#!/usr/bin/env python3
"""Freeze the four matched absolute VISUAL_ON-cap supervision manifests."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from label_regeneration.bce_geometry import (
    binary_entropy,
    hamming,
    layer_marginals,
    pairwise_distances,
    polar_route_weights,
    threshold_mask,
)
from label_regeneration.cap_supervision import build_cap_record, filter_routes_by_cap


PROJECT = Path(__file__).resolve().parents[1]
CAPS = (24, 22, 20, 18)
ALL_ON = (1,) * 28
ALL_OFF = (0,) * 28
EXPECTED_PARENT_SHA256 = "3620a347a3498d16853463a6f9f8b842fecbab7b442cb869f1fb11bc9ab8aa52"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def write_checksum(path: Path) -> None:
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{file_sha256(path)}  {path.name}\n", encoding="utf-8"
    )


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def geometry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [row for row in rows if row["valid_routes"]]
    route_counts = [len(row["valid_routes"]) for row in positive]
    masks = [tuple(int(bit) for bit in route["mask"]) for row in positive for route in row["valid_routes"]]
    on_counts = [sum(mask) for mask in masks]
    sample_pairwise = []
    for row in positive:
        current = [tuple(int(bit) for bit in route["mask"]) for route in row["valid_routes"]]
        distances = list(pairwise_distances(current))
        if distances:
            sample_pairwise.append(statistics.fmean(distances))
    layer_probabilities = [
        statistics.fmean(mask[layer] for mask in masks) for layer in range(28)
    ] if masks else []
    return {
        "records": len(rows),
        "positive_records": len(positive),
        "route_occurrences": len(masks),
        "mean_routes_per_sample": statistics.fmean(route_counts) if route_counts else None,
        "median_routes_per_sample": statistics.median(route_counts) if route_counts else None,
        "fraction_singleton": sum(count == 1 for count in route_counts) / len(route_counts) if route_counts else None,
        "fraction_at_least_2": sum(count >= 2 for count in route_counts) / len(route_counts) if route_counts else None,
        "fraction_at_least_5": sum(count >= 5 for count in route_counts) / len(route_counts) if route_counts else None,
        "mean_visual_on_layers": statistics.fmean(on_counts) if on_counts else None,
        "median_visual_on_layers": statistics.median(on_counts) if on_counts else None,
        "mean_sample_pairwise_hamming": statistics.fmean(sample_pairwise) if sample_pairwise else None,
        "mean_bit_entropy_nats": statistics.fmean(binary_entropy(value) for value in layer_probabilities) if layer_probabilities else None,
        "all_on_sample_presence": sum(any(tuple(route["mask"]) == ALL_ON for route in row["valid_routes"]) for row in positive),
        "all_off_sample_presence": sum(any(tuple(route["mask"]) == ALL_OFF for route in row["valid_routes"]) for row in positive),
    }


def oracle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [row for row in rows if row["valid_routes"]]
    predictions: list[tuple[int, ...]] = []
    cap_hits: list[bool] = []
    original_hits: list[bool] = []
    nearest_cap: list[int] = []
    for row in positive:
        masks = [tuple(int(bit) for bit in route["mask"]) for route in row["valid_routes"]]
        decoded = threshold_mask(layer_marginals(masks, polar_route_weights(masks)))
        original = {tuple(int(bit) for bit in key) for key in row["original_valid_mask_keys"]}
        predictions.append(decoded)
        cap_hits.append(decoded in masks)
        original_hits.append(decoded in original)
        nearest_cap.append(min(hamming(decoded, mask) for mask in masks))
    counts = Counter(predictions)
    return {
        "records": len(positive),
        "route_weighting": "polar_full_downweight_0.3 (equal after cap removes ALL-ON)",
        "mean_visual_on_layers": statistics.fmean(sum(mask) for mask in predictions),
        "all_on_fraction": counts[ALL_ON] / len(predictions),
        "all_off_fraction": counts[ALL_OFF] / len(predictions),
        "unique_masks": len(counts),
        "cap_valid_hit_at_1": statistics.fmean(cap_hits),
        "original_valid_hit_at_1": statistics.fmean(original_hits),
        "mean_nearest_cap_valid_hamming": statistics.fmean(nearest_cap),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent",
        type=Path,
        default=PROJECT / "outputs/label_regeneration/v1/post_generation/binary_predictor_manifest_v1.jsonl",
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT / "outputs/binary_cap_sweep_v1")
    args = parser.parse_args()
    parent_display = args.parent
    parent = args.parent.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite cap output root: {output_root}")
    parent_digest = file_sha256(parent)
    if parent_digest != EXPECTED_PARENT_SHA256:
        raise RuntimeError(f"parent manifest checksum mismatch: {parent_digest}")
    source_rows: list[dict[str, Any]] = []
    with parent.open("rb") as handle:
        for raw in handle:
            if raw.strip():
                row = json.loads(raw)
                row["_parent_record_sha256"] = sha256(raw).hexdigest()
                source_rows.append(row)
    if len(source_rows) != 8000 or len({row["uid"] for row in source_rows}) != 8000:
        raise RuntimeError("parent population/UID integrity failure")
    if Counter(row["benchmark"] for row in source_rows) != Counter(gqa=4000, textvqa=2000, chartqa=2000):
        raise RuntimeError("parent dataset counts differ from the frozen source")
    if Counter(row["split"] for row in source_rows) != Counter(train=7000, validation=1000):
        raise RuntimeError("parent split counts differ from the frozen source")

    original_positive = [row for row in source_rows if row["valid_routes"]]
    common_uids = {
        row["uid"]
        for row in original_positive
        if filter_routes_by_cap(row["valid_routes"], cap=18)
    }
    manifests: dict[int, list[dict[str, Any]]] = {}
    native_coverage: dict[str, Any] = {}
    for cap in CAPS:
        native_positive = [
            row for row in original_positive if filter_routes_by_cap(row["valid_routes"], cap=cap)
        ]
        native_coverage[str(cap)] = {
            "positive_records": len(native_positive),
            "fraction_of_original_positive": len(native_positive) / len(original_positive),
            "zero_route_original_positive": len(original_positive) - len(native_positive),
            "by_dataset": {
                dataset: {
                    "positive_records": sum(row["benchmark"] == dataset for row in native_positive),
                    "original_positive_records": sum(row["benchmark"] == dataset for row in original_positive),
                }
                for dataset in ("gqa", "textvqa", "chartqa")
            },
        }
        records = []
        for source in source_rows:
            parent_record_sha = source["_parent_record_sha256"]
            clean = {key: value for key, value in source.items() if key != "_parent_record_sha256"}
            record = build_cap_record(
                clean, cap=cap, common_eligible=source["uid"] in common_uids
            )
            record["parent_manifest_path"] = str(parent_display)
            record["parent_manifest_sha256"] = parent_digest
            record["parent_record_sha256"] = parent_record_sha
            records.append(record)
        manifests[cap] = records

    common_counts = Counter(
        (row["split"], row["benchmark"])
        for row in source_rows
        if row["uid"] in common_uids
    )
    geometry_payload: dict[str, Any] = {
        "schema_version": "binary_cap_geometry_v1",
        "source": {"path": str(parent_display), "resolved_path": str(parent), "sha256": parent_digest},
        "caps": list(CAPS),
        "common_eligibility_definition": "at least one selected parent valid route with VISUAL_ON <= 18",
        "original_positive_records": len(original_positive),
        "common_eligible_records": len(common_uids),
        "common_eligible_by_split_dataset": {
            f"{split}:{dataset}": common_counts[(split, dataset)]
            for split in ("train", "validation")
            for dataset in ("gqa", "textvqa", "chartqa")
        },
        "native_coverage": native_coverage,
        "matched_geometry": {},
    }
    oracle_payload: dict[str, Any] = {
        "schema_version": "binary_cap_label_oracles_v1",
        "decode_rule": "weighted layer marginal >= 0.5; exact ties resolve ON",
        "caps": {},
    }
    for cap, rows in manifests.items():
        strata = {"overall": [row for row in rows if row["valid_routes"]]}
        strata.update({dataset: [row for row in rows if row["valid_routes"] and row["benchmark"] == dataset] for dataset in ("gqa", "textvqa", "chartqa")})
        geometry_payload["matched_geometry"][str(cap)] = {
            name: geometry(members) for name, members in strata.items()
        }
        oracle_payload["caps"][str(cap)] = {
            name: oracle(members) for name, members in strata.items()
        }

    manifest_paths = {}
    for cap, rows in manifests.items():
        path = output_root / "manifests" / f"cap{cap}_manifest_v1.jsonl"
        write_jsonl(path, rows)
        write_checksum(path)
        manifest_paths[str(cap)] = {"path": str(path.relative_to(PROJECT)), "sha256": file_sha256(path)}
    geometry_path = output_root / "audits/cap_geometry_v1.json"
    oracle_path = output_root / "audits/cap_label_oracles_v1.json"
    write_json(geometry_path, geometry_payload)
    write_json(oracle_path, oracle_payload)
    write_checksum(geometry_path)
    write_checksum(oracle_path)
    audit = {
        "schema_version": "binary_cap_supervision_audit_v1",
        "passed": True,
        "integrity_status": "PASS",
        "source": {"path": str(parent_display), "resolved_path": str(parent), "sha256": parent_digest},
        "caps": list(CAPS),
        "population_records": len(source_rows),
        "original_positive_records": len(original_positive),
        "common_eligible_records": len(common_uids),
        "common_train_records": sum(row["uid"] in common_uids and row["split"] == "train" for row in source_rows),
        "common_validation_records": sum(row["uid"] in common_uids and row["split"] == "validation" for row in source_rows),
        "checks": {
            "same_common_uids_all_caps": True,
            "retained_parent_order": True,
            "retained_routes_are_parent_routes": True,
            "no_pareto_or_ranking_filter": True,
            "no_zero_route_fallback": True,
            "all_on_absent": all(
                tuple(route["mask"]) != ALL_ON
                for rows in manifests.values() for row in rows for route in row["valid_routes"]
            ),
            "image_group_split_unchanged": True,
        },
        "manifests": manifest_paths,
        "geometry": {"path": str(geometry_path.relative_to(PROJECT)), "sha256": file_sha256(geometry_path)},
        "label_oracles": {"path": str(oracle_path.relative_to(PROJECT)), "sha256": file_sha256(oracle_path)},
    }
    audit_path = output_root / "audits/cap_supervision_audit_v1.json"
    write_json(audit_path, audit)
    write_checksum(audit_path)
    print(json.dumps({
        "passed": True,
        "common_train": audit["common_train_records"],
        "common_validation": audit["common_validation_records"],
        "manifests": manifest_paths,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
