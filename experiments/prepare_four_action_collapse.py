#!/usr/bin/env python3
"""Freeze mandatory-boundary metadata and the collapse-pilot population."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.train_binary_polar import file_sha256
from four_action_online_router.data import (
    load_verified_manifest,
    mandatory_boundary_record,
    select_boundary_pilot,
)


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def write_frozen(path: Path, content: str) -> None:
    """Atomically create a deterministic artifact or verify the existing copy."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"frozen-artifact lock already exists: {lock}") from exc
    os.close(descriptor)
    temporary: Path | None = None
    try:
        if path.exists():
            if path.read_text(encoding="utf-8") != content:
                raise FileExistsError(f"refusing to overwrite different artifact: {path}")
            return
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", delete=False
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


def boundary_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("boundary audit requires records")
    action_sets = Counter("+".join(row["valid_nonfull_actions"]) for row in rows)
    by_dataset = {}
    for dataset in ("gqa", "chartqa", "textvqa"):
        selected = [row for row in rows if row["dataset"] == dataset]
        by_dataset[dataset] = {
            "records": len(selected),
            "singleton": sum(bool(row["singleton"]) for row in selected),
            "boundary_layer_counts": dict(
                sorted(Counter(int(row["boundary_layer"]) for row in selected).items())
            ),
            "valid_action_set_counts": dict(
                sorted(Counter("+".join(row["valid_nonfull_actions"]) for row in selected).items())
            ),
        }
    return {
        "records": len(rows),
        "unique_uids": len({row["uid"] for row in rows}),
        "full_invalid_records": sum(
            "FULL" not in row["valid_nonfull_actions"] for row in rows
        ),
        "nonfull_valid_records": sum(bool(row["valid_nonfull_actions"]) for row in rows),
        "singleton_records": sum(bool(row["singleton"]) for row in rows),
        "boundary_layer": {
            "min": min(int(row["boundary_layer"]) for row in rows),
            "max": max(int(row["boundary_layer"]) for row in rows),
            "mean": sum(int(row["boundary_layer"]) for row in rows) / len(rows),
            "counts": dict(
                sorted(Counter(int(row["boundary_layer"]) for row in rows).items())
            ),
        },
        "valid_action_set_counts": dict(sorted(action_sets.items())),
        "by_dataset": by_dataset,
    }


def render_boundary_audit(
    *, audit: dict[str, Any], manifest_path: Path, manifest_sha256: str,
    boundary_sha256: str, pilot_sha256: str, plan_sha256: str,
) -> str:
    lines = [
        "# Mandatory-Boundary Audit",
        "",
        "## Frozen provenance",
        "",
        f"- Source manifest: `{manifest_path}`",
        f"- Source manifest SHA-256: `{manifest_sha256}`",
        f"- Boundary manifest SHA-256: `{boundary_sha256}`",
        f"- Pilot subset SHA-256: `{pilot_sha256}`",
        f"- Plan SHA-256: `{plan_sha256}`",
        "",
        "## Integrity gates",
        "",
        f"- W2C records: {audit['records']}",
        f"- Unique UIDs: {audit['unique_uids']}",
        f"- FULL invalid at boundary: {audit['full_invalid_records']}/{audit['records']}",
        f"- At least one non-FULL valid action: {audit['nonfull_valid_records']}/{audit['records']}",
        f"- Singleton boundaries: {audit['singleton_records']}/{audit['records']}",
        "",
        "## Boundary distribution",
        "",
        f"- Mean layer: {audit['boundary_layer']['mean']:.3f}",
        f"- Range: {audit['boundary_layer']['min']}–{audit['boundary_layer']['max']}",
        f"- Valid action sets: `{json.dumps(audit['valid_action_set_counts'], sort_keys=True)}`",
        "",
        "## Dataset counts",
        "",
        "| Dataset | Records | Singleton |",
        "|---|---:|---:|",
    ]
    for dataset, values in audit["by_dataset"].items():
        lines.append(f"| {dataset} | {values['records']} | {values['singleton']} |")
    lines.extend(
        [
            "",
            "All required A0 invariants pass. The canonical artifacts are under",
            "`analysis/4action_collapse/`; the legacy A0 paths under",
            "`analysis/4action_router/` may link to these frozen files.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--plan", default="plans/four_action_collapse.md")
    parser.add_argument("--output-dir", default="analysis/4action_collapse")
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--w2c-per-dataset", type=int, default=32)
    parser.add_argument("--c2c-per-dataset", type=int, default=8)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    rows = load_verified_manifest(manifest_path, args.manifest_sha256)
    train_rows = [row for row in rows if row.get("split") == "train"]
    boundaries = [
        mandatory_boundary_record(row, num_layers=args.num_layers)
        for row in train_rows if row.get("route_type") == "W2C"
    ]
    boundaries.sort(key=lambda row: row["uid"])
    audit = boundary_audit(boundaries)
    if not (
        audit["records"] == audit["unique_uids"]
        == audit["full_invalid_records"] == audit["nonfull_valid_records"]
    ):
        raise RuntimeError("mandatory-boundary integrity gate failed")

    pilot = select_boundary_pilot(
        train_rows,
        boundaries,
        w2c_per_dataset=args.w2c_per_dataset,
        c2c_per_dataset=args.c2c_per_dataset,
        seed=args.seed,
        num_layers=args.num_layers,
    )
    by_uid = {row["uid"]: row for row in boundaries}
    pilot["schema_version"] = "four_action_mandatory_boundary_pilot_v1"
    pilot["source_manifest"] = str(manifest_path)
    pilot["source_manifest_sha256"] = args.manifest_sha256
    plan_path = Path(args.plan)
    pilot["plan"] = str(plan_path)
    pilot["plan_sha256"] = file_sha256(plan_path)
    pilot["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    pilot["w2c_records"] = [
        {
            **by_uid[uid],
            "teacher_route_index": min(by_uid[uid]["boundary_route_indices"]),
        }
        for uid in pilot.pop("w2c_uids")
    ]
    pilot["c2c_records"] = [{"uid": uid} for uid in pilot.pop("c2c_uids")]

    output_dir = Path(args.output_dir)
    boundary_path = output_dir / "boundary_manifest.jsonl"
    pilot_path = output_dir / "pilot_subset.json"
    write_frozen(boundary_path, _jsonl_text(boundaries))
    write_frozen(pilot_path, _json_text(pilot))
    boundary_sha = file_sha256(boundary_path)
    pilot_sha = file_sha256(pilot_path)
    write_frozen(
        boundary_path.with_suffix(".jsonl.sha256"),
        f"{boundary_sha}  {boundary_path.name}\n",
    )
    write_frozen(
        pilot_path.with_suffix(".json.sha256"), f"{pilot_sha}  {pilot_path.name}\n"
    )
    write_frozen(
        output_dir / "boundary_audit.md",
        render_boundary_audit(
            audit=audit,
            manifest_path=manifest_path,
            manifest_sha256=args.manifest_sha256,
            boundary_sha256=boundary_sha,
            pilot_sha256=pilot_sha,
            plan_sha256=file_sha256(plan_path),
        ),
    )
    write_frozen(
        output_dir / "boundary_audit.json",
        _json_text(
            {
                "schema_version": "four_action_mandatory_boundary_audit_v1",
                "source_manifest_sha256": args.manifest_sha256,
                "boundary_manifest_sha256": boundary_sha,
                "pilot_subset_sha256": pilot_sha,
                "plan_sha256": file_sha256(plan_path),
                **audit,
            }
        ),
    )
    print(
        json.dumps(
            {
                "event": "four_action_boundary_preparation_complete",
                "records": audit["records"],
                "boundary_manifest_sha256": boundary_sha,
                "pilot_subset_sha256": pilot_sha,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
