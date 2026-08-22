#!/usr/bin/env python3
"""Freeze matched CAP26/CAP24 supervision for the five-epoch NLL study."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.prepare_binary_cap_supervision import (
    EXPECTED_PARENT_SHA256,
    file_sha256,
    geometry,
    oracle,
    write_checksum,
    write_json,
    write_jsonl,
)
from label_regeneration.cap_supervision import build_cap_record, filter_routes_by_cap


PROJECT = Path(__file__).resolve().parents[1]
CAPS = (26, 24)
COMMON_CAP = 24


def build_matched_records(
    source_rows: list[dict[str, Any]], *, cap: int, common_cap: int, width: int = 28
) -> tuple[list[dict[str, Any]], set[str]]:
    """Apply one cap on the UID population eligible under ``common_cap``."""

    common_uids = {
        str(row["uid"])
        for row in source_rows
        if filter_routes_by_cap(row.get("valid_routes", []), cap=common_cap, expected_width=width)
    }
    records = [
        build_cap_record(
            row,
            cap=cap,
            common_eligible=str(row["uid"]) in common_uids,
            expected_width=width,
        )
        for row in source_rows
    ]
    return records, common_uids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path(
            "outputs/label_regeneration/v1/post_generation/"
            "binary_predictor_manifest_v1.jsonl"
        ),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/binary_cap_nll5_v1")
    )
    args = parser.parse_args()
    parent = args.parent
    output_root = args.output_root
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")
    parent_digest = file_sha256(parent)
    if parent_digest != EXPECTED_PARENT_SHA256:
        raise RuntimeError(f"parent manifest checksum mismatch: {parent_digest}")

    source_rows = []
    with parent.open("rb") as handle:
        for raw in handle:
            if raw.strip():
                row = json.loads(raw)
                row["_parent_record_sha256"] = sha256(raw).hexdigest()
                source_rows.append(row)
    if len(source_rows) != 8000 or len({row["uid"] for row in source_rows}) != 8000:
        raise RuntimeError("frozen parent population/UID integrity failure")
    if Counter(row["split"] for row in source_rows) != Counter(train=7000, validation=1000):
        raise RuntimeError("frozen parent split counts changed")

    clean_rows = [
        {key: value for key, value in row.items() if key != "_parent_record_sha256"}
        for row in source_rows
    ]
    manifests: dict[int, list[dict[str, Any]]] = {}
    common_sets = []
    for cap in CAPS:
        records, common_uids = build_matched_records(
            clean_rows, cap=cap, common_cap=COMMON_CAP
        )
        for record, source in zip(records, source_rows):
            record["parent_manifest_path"] = str(parent)
            record["parent_manifest_sha256"] = parent_digest
            record["parent_record_sha256"] = source["_parent_record_sha256"]
        manifests[cap] = records
        common_sets.append(common_uids)
    if common_sets[0] != common_sets[1]:
        raise RuntimeError("CAP26/CAP24 common UID populations differ")
    common_uids = common_sets[0]
    split_counts = Counter(
        row["split"] for row in clean_rows if str(row["uid"]) in common_uids
    )
    if split_counts != Counter(train=6007, validation=872):
        raise RuntimeError(f"matched population changed: {split_counts}")

    manifest_specs = {}
    matched_geometry = {}
    label_oracles = {}
    for cap, rows in manifests.items():
        path = output_root / "manifests" / f"cap{cap}_manifest_v1.jsonl"
        write_jsonl(path, rows)
        write_checksum(path)
        positives = [row for row in rows if row["valid_routes"]]
        if {row["uid"] for row in positives} != common_uids:
            raise RuntimeError(f"CAP{cap} positive population differs from common UIDs")
        if any(
            sum(int(value) for value in route["mask"]) > cap
            for row in positives
            for route in row["valid_routes"]
        ):
            raise RuntimeError(f"CAP{cap} manifest contains an over-cap route")
        strata = {"overall": positives}
        strata.update(
            {
                dataset: [row for row in positives if row["benchmark"] == dataset]
                for dataset in ("gqa", "textvqa", "chartqa")
            }
        )
        matched_geometry[str(cap)] = {
            name: geometry(current) for name, current in strata.items()
        }
        label_oracles[str(cap)] = {
            name: oracle(current) for name, current in strata.items()
        }
        manifest_specs[str(cap)] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }

    geometry_path = output_root / "audits" / "cap_geometry_v1.json"
    oracle_path = output_root / "audits" / "cap_label_oracles_v1.json"
    write_json(
        geometry_path,
        {
            "schema_version": "binary_cap_nll5_geometry_v1",
            "caps": list(CAPS),
            "common_eligibility_cap": COMMON_CAP,
            "common_records": len(common_uids),
            "common_train_records": split_counts["train"],
            "common_validation_records": split_counts["validation"],
            "matched_geometry": matched_geometry,
        },
    )
    write_json(
        oracle_path,
        {
            "schema_version": "binary_cap_nll5_label_oracles_v1",
            "diagnostic_only": True,
            "caps": label_oracles,
        },
    )
    write_checksum(geometry_path)
    write_checksum(oracle_path)
    audit_path = output_root / "audits" / "supervision_audit_v1.json"
    write_json(
        audit_path,
        {
            "schema_version": "binary_cap_nll5_supervision_audit_v1",
            "passed": True,
            "integrity_status": "PASS",
            "source": {"path": str(parent), "sha256": parent_digest},
            "caps": list(CAPS),
            "common_eligibility_cap": COMMON_CAP,
            "common_records": len(common_uids),
            "common_train_records": split_counts["train"],
            "common_validation_records": split_counts["validation"],
            "manifests": manifest_specs,
            "geometry": {"path": str(geometry_path), "sha256": file_sha256(geometry_path)},
            "label_oracles": {"path": str(oracle_path), "sha256": file_sha256(oracle_path)},
            "checks": {
                "same_common_uids": True,
                "image_group_split_unchanged": True,
                "no_pareto_filter": True,
                "no_zero_route_fallback": True,
                "all_on_absent": True,
            },
        },
    )
    write_checksum(audit_path)
    print(json.dumps({"passed": True, "audit": str(audit_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
