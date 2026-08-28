#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from tools.research_analysis.four_action.sequential_label_jobs import (
    file_sha256,
    select_sequential_smoke,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def encoded_jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )


def write_or_verify(path: Path, content: str, *, resume: bool) -> None:
    if path.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {path}")
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"existing artifact differs from deterministic rebuild: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the fixed eight-sample sequential smoke.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sequential_four_action_label_conversion.yaml"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = Path(config["source_manifest"])
    selected, coverage = select_sequential_smoke(
        read_jsonl(source), config["smoke_uids"]
    )
    output_root = Path(config["output_root"]) / "smoke"
    manifest = output_root / "smoke_manifest_v1.jsonl"
    write_or_verify(manifest, encoded_jsonl(selected), resume=args.resume)
    report = {
        "schema_version": "exact_sequential_smoke_selection_v1",
        "source_manifest": str(source.resolve()),
        "source_manifest_sha256": file_sha256(source),
        "smoke_manifest": str(manifest.resolve()),
        "smoke_manifest_sha256": file_sha256(manifest),
        "selected_uids": [row["uid"] for row in selected],
        "coverage": coverage,
        "samples": [
            {
                "uid": row["uid"],
                "dataset": row["dataset"],
                "source_status": row["source_current_all_on_status"],
                "source_positive_route_count": row["source_positive_route_count"],
                "source_off_count_min": min(
                    int(route["source_off_count"])
                    for route in row["source_positive_routes"]
                ),
                "source_off_count_max": max(
                    int(route["source_off_count"])
                    for route in row["source_positive_routes"]
                ),
                "contains_all_off_seed": any(
                    bool(route["source_all_off"])
                    for route in row["source_positive_routes"]
                ),
            }
            for row in selected
        ],
    }
    analysis_path = Path(config["analysis_root"]) / "smoke_selection_v1.json"
    write_or_verify(
        analysis_path,
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        resume=args.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
