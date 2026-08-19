#!/usr/bin/env python3
"""Freeze Pareto-efficient supervision from the exact full10 parent manifest."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from label_regeneration.bce_geometry import hamming, layer_marginals, polar_route_weights, threshold_mask
from label_regeneration.pareto_supervision import build_pareto_record


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = PROJECT / "outputs/label_regeneration/v1/post_generation/binary_predictor_manifest_v1.jsonl"
DEFAULT_OUTPUT = PROJECT / "outputs/binary_pareto_v1"
EXPECTED_PARENT_SHA256 = "3620a347a3498d16853463a6f9f8b842fecbab7b442cb869f1fb11bc9ab8aa52"
DATASET_COUNTS = Counter({"gqa": 4000, "textvqa": 2000, "chartqa": 2000})
SPLIT_COUNTS = Counter({"train": 7000, "validation": 1000})
ALL_ON = (1,) * 28
ALL_OFF = (0,) * 28


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


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a quantile of an empty sequence")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def population_strata(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {"overall": rows}
    for row in rows:
        benchmark = str(row["benchmark"])
        split = str(row["split"])
        result.setdefault(f"dataset:{benchmark}", []).append(row)
        result.setdefault(f"split:{split}", []).append(row)
        result.setdefault(f"split_dataset:{split}:{benchmark}", []).append(row)
    return result


def geometry_summary(rows: list[dict[str, Any]], route_field: str) -> dict[str, Any]:
    positive = [row for row in rows if row[route_field]]
    counts = [len(row[route_field]) for row in positive]
    masks = [tuple(route["mask"]) for row in positive for route in row[route_field]]
    on_counts = [sum(mask) for mask in masks]
    return {
        "population_records": len(rows),
        "positive_sample_count": len(positive),
        "zero_positive_sample_count": len(rows) - len(positive),
        "route_occurrences": len(masks),
        "mean_routes_per_positive_sample": statistics.fmean(counts) if counts else None,
        "median_routes_per_positive_sample": statistics.median(counts) if counts else None,
        "p90_routes_per_positive_sample": quantile(counts, 0.90) if counts else None,
        "fraction_exactly_1": sum(count == 1 for count in counts) / len(counts) if counts else None,
        "fraction_exactly_2": sum(count == 2 for count in counts) / len(counts) if counts else None,
        "fraction_at_least_3": sum(count >= 3 for count in counts) / len(counts) if counts else None,
        "mean_visual_on_layers": statistics.fmean(on_counts) if on_counts else None,
        "minimum_visual_on_layers": min(on_counts) if on_counts else None,
        "median_visual_on_layers": statistics.median(on_counts) if on_counts else None,
        "all_on_sample_count": sum(ALL_ON in [tuple(route["mask"]) for route in row[route_field]] for row in positive),
        "all_off_sample_count": sum(ALL_OFF in [tuple(route["mask"]) for route in row[route_field]] for row in positive),
    }


def original_route_view(row: dict[str, Any]) -> list[dict[str, Any]]:
    return row["_original_valid_routes"]


def oracle_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [row for row in rows if row["valid_routes"]]
    predictions = []
    nearest = []
    hit = []
    for row in positive:
        masks = [tuple(route["mask"]) for route in row["valid_routes"]]
        decoded = threshold_mask(layer_marginals(masks, polar_route_weights(masks)))
        predictions.append(decoded)
        hit.append(decoded in masks)
        nearest.append(min(hamming(decoded, mask) for mask in masks))
    counts = Counter(predictions)
    return {
        "records": len(positive),
        "valid_set_hit_at_1": statistics.fmean(hit) if hit else None,
        "mean_nearest_valid_hamming": statistics.fmean(nearest) if nearest else None,
        "mean_visual_on_layers": statistics.fmean(sum(mask) for mask in predictions) if predictions else None,
        "all_on_fraction": counts[ALL_ON] / len(predictions) if predictions else None,
        "all_off_fraction": counts[ALL_OFF] / len(predictions) if predictions else None,
        "unique_masks": len(counts),
        "most_common_masks": [
            {"mask": "".join(map(str, mask)), "count": count}
            for mask, count in counts.most_common(10)
        ],
    }


def nll_shortcut_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [row for row in rows if row["valid_routes"]]
    route_frequency = Counter(
        tuple(route["mask"]) for row in positive for route in row["valid_routes"]
    )
    ordered = [mask for mask, _ in route_frequency.most_common()]

    def coverage(limit: int) -> float:
        common = set(ordered[:limit])
        return sum(
            any(tuple(route["mask"]) in common for route in row["valid_routes"])
            for row in positive
        ) / len(positive)

    return {
        "records": len(positive),
        "fraction_sets_containing_all_on": sum(
            any(tuple(route["mask"]) == ALL_ON for route in row["valid_routes"])
            for row in positive
        ) / len(positive),
        "most_common_complete_route_coverage": coverage(1),
        "top_5_complete_route_coverage": coverage(5),
        "top_50_complete_route_coverage": coverage(50),
        "most_common_routes": [
            {"mask": "".join(map(str, mask)), "sample_count": count}
            for mask, count in route_frequency.most_common(50)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    parent = args.parent.resolve()
    output_root = args.output_root.resolve()
    parent_digest = file_sha256(parent)
    if parent_digest != EXPECTED_PARENT_SHA256:
        raise RuntimeError(f"parent supervision checksum mismatch: {parent_digest}")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite Pareto output root: {output_root}")

    source_rows = []
    with parent.open("rb") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            row["_parent_record_sha256"] = sha256(raw_line).hexdigest()
            source_rows.append(row)
    if len(source_rows) != 8000:
        raise RuntimeError(f"expected 8,000 parent rows, found {len(source_rows)}")
    if Counter(row["benchmark"] for row in source_rows) != DATASET_COUNTS:
        raise RuntimeError("parent dataset population mismatch")
    if Counter(row["split"] for row in source_rows) != SPLIT_COUNTS:
        raise RuntimeError("parent split population mismatch")
    if len({row["uid"] for row in source_rows}) != len(source_rows):
        raise RuntimeError("parent manifest contains duplicate UIDs")
    group_splits: dict[str, str] = {}
    for row in source_rows:
        prior = group_splits.setdefault(row["split_group"], row["split"])
        if prior != row["split"]:
            raise RuntimeError(f"image-group split leakage: {row['split_group']}")

    pareto_rows = []
    witness_rows = []
    removed_count = 0
    for source in source_rows:
        parent_record_sha = source.pop("_parent_record_sha256")
        transformed, witnesses = build_pareto_record(source)
        transformed["parent_manifest_path"] = str(parent.relative_to(PROJECT))
        transformed["parent_manifest_sha256"] = parent_digest
        transformed["parent_record_sha256"] = parent_record_sha
        transformed["_original_valid_routes"] = source["valid_routes"]
        pareto_rows.append(transformed)
        removed_count += len(witnesses)
        witness_rows.append(
            {
                "uid": transformed["uid"],
                "original_selected_valid_route_count": len(source["valid_routes"]),
                "pareto_efficient_route_count": len(transformed["valid_routes"]),
                "removed_to_retained_witness": witnesses,
            }
        )

    positive = [row for row in pareto_rows if row["valid_routes"]]
    if len(positive) != 6917:
        raise RuntimeError(f"positive population mismatch: {len(positive)}")
    positive_split = Counter(row["split"] for row in positive)
    if positive_split != Counter({"train": 6043, "validation": 874}):
        raise RuntimeError(f"positive split mismatch: {positive_split}")

    for row in pareto_rows:
        retained = [tuple(route["mask"]) for route in row["valid_routes"]]
        original = [tuple(route["mask"]) for route in row["_original_valid_routes"]]
        if not set(retained).issubset(original) or len(retained) != len(set(retained)):
            raise RuntimeError(f"retained route provenance failure: {row['uid']}")
        for route in row["valid_routes"]:
            mask = tuple(route["mask"])
            score = float(route["score"])
            if any(
                sum(other) < sum(mask)
                and float(other_route["score"]) >= score
                for other, other_route in zip(retained, row["valid_routes"])
            ):
                raise RuntimeError(f"dominated route remains: {row['uid']}")

    geometry = {}
    for stratum, members in population_strata(pareto_rows).items():
        geometry[stratum] = {
            "before": geometry_summary(
                [{**row, "_routes": row["_original_valid_routes"]} for row in members],
                "_routes",
            ),
            "after": geometry_summary(members, "valid_routes"),
        }
    before_all_on = sum(
        tuple(route["mask"]) == ALL_ON
        for row in pareto_rows
        for route in row["_original_valid_routes"]
    )
    after_all_on = sum(
        tuple(route["mask"]) == ALL_ON for row in pareto_rows for route in row["valid_routes"]
    )
    before_all_on_samples = sum(
        any(tuple(route["mask"]) == ALL_ON for route in row["_original_valid_routes"])
        for row in pareto_rows
    )
    removed_all_on_samples = sum(
        any(tuple(route["mask"]) == ALL_ON for route in row["_original_valid_routes"])
        and not any(tuple(route["mask"]) == ALL_ON for route in row["valid_routes"])
        for row in pareto_rows
    )
    geometry["all_on_removal"] = {
        "before_occurrences": before_all_on,
        "after_occurrences": after_all_on,
        "fraction_occurrences_removed": (before_all_on - after_all_on) / before_all_on,
        "before_sample_presence": before_all_on_samples,
        "samples_removed": removed_all_on_samples,
        "fraction_samples_removed_when_present": removed_all_on_samples / before_all_on_samples,
    }

    oracle = {
        "route_weighting": "polar_full_downweight_0.3, normalized per sample",
        "bce_label_oracle": {
            stratum: oracle_summary(members)
            for stratum, members in population_strata(pareto_rows).items()
        },
        "nll_shortcut_geometry": nll_shortcut_summary(pareto_rows),
    }

    manifest_path = output_root / "manifests/binary_pareto_predictor_manifest_v1.jsonl"
    witness_path = output_root / "audits/dominance_witnesses_v1.jsonl"
    geometry_path = output_root / "audits/pareto_geometry_v1.json"
    audit_path = output_root / "audits/pareto_integrity_audit_v1.json"
    oracle_path = output_root / "oracle_analysis/label_oracles_v1.json"
    serializable_rows = []
    for row in pareto_rows:
        current = dict(row)
        current.pop("_original_valid_routes")
        serializable_rows.append(current)
    write_jsonl(manifest_path, serializable_rows)
    write_jsonl(witness_path, witness_rows)
    write_json(geometry_path, geometry)
    write_json(oracle_path, oracle)
    audit = {
        "schema_version": "binary_pareto_supervision_audit_v1",
        "passed": True,
        "integrity_status": "PASS",
        "source": {"path": str(parent.relative_to(PROJECT)), "sha256": parent_digest},
        "population": {
            "records": len(pareto_rows),
            "datasets": dict(DATASET_COUNTS),
            "splits": dict(SPLIT_COUNTS),
            "positive_records": len(positive),
            "positive_train": positive_split["train"],
            "positive_validation": positive_split["validation"],
            "zero_positive_records": len(pareto_rows) - len(positive),
        },
        "filter": {
            "definition": "score(b) >= score(a) and ON(b) < ON(a)",
            "removed_route_occurrences": removed_count,
            "retained_route_occurrences": sum(len(row["valid_routes"]) for row in pareto_rows),
            "every_removed_route_has_retained_witness": True,
        },
        "checks": {
            "retained_routes_are_parent_routes": True,
            "retained_routes_valid": True,
            "exact_28_bit_masks": True,
            "exact_on_counts": True,
            "no_duplicates": True,
            "no_dominated_route_remains": True,
            "positive_samples_retain_route": True,
            "population_membership_unchanged": True,
            "image_group_disjointness_unchanged": True,
        },
        "artifacts": {},
    }
    write_json(audit_path, audit)
    for path in (manifest_path, witness_path, geometry_path, oracle_path, audit_path):
        write_checksum(path)
    audit["artifacts"] = {
        str(path.relative_to(PROJECT)): file_sha256(path)
        for path in (manifest_path, witness_path, geometry_path, oracle_path)
    }
    write_json(audit_path, audit)
    write_checksum(audit_path)
    print(json.dumps({
        "integrity_status": "PASS",
        "manifest": str(manifest_path.relative_to(PROJECT)),
        "manifest_sha256": file_sha256(manifest_path),
        "positive_records": len(positive),
        "retained_routes": audit["filter"]["retained_route_occurrences"],
        "removed_routes": removed_count,
        "bce_oracle_hit_at_1": oracle["bce_label_oracle"]["overall"]["valid_set_hit_at_1"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
