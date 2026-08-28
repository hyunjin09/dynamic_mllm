#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from tools.research_analysis.four_action.sequential_label_jobs import file_sha256


def record_hashes(root: Path) -> dict[str, str]:
    return {
        path.name: file_sha256(path)
        for path in sorted((root / "records").glob("*.json"))
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Snapshot or verify exact smoke resume.")
    parser.add_argument("--mode", choices=("snapshot", "verify"), required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sequential_four_action_label_conversion.yaml"),
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(config["output_root"]) / "smoke"
    analysis = Path(config["analysis_root"])
    snapshot_path = analysis / "smoke_pre_resume_hashes_v1.json"
    verification_path = analysis / "smoke_resume_verification_v1.json"
    hashes = record_hashes(root)
    if len(hashes) != 8:
        raise RuntimeError(f"expected 8 smoke records before resume check, found {len(hashes)}")
    if args.mode == "snapshot":
        if snapshot_path.exists():
            existing = json.loads(snapshot_path.read_text(encoding="utf-8"))
            if existing.get("record_sha256") != hashes:
                raise RuntimeError("existing pre-resume snapshot differs from current records")
        else:
            analysis.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_text(
                json.dumps(
                    {
                        "schema_version": "exact_sequential_smoke_pre_resume_hashes_v1",
                        "record_sha256": hashes,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        return 0

    if not snapshot_path.is_file():
        raise FileNotFoundError(snapshot_path)
    before = json.loads(snapshot_path.read_text(encoding="utf-8"))["record_sha256"]
    report = {
        "schema_version": "exact_sequential_smoke_resume_verification_v1",
        "passed": hashes == before,
        "record_count": len(hashes),
        "records_unchanged": hashes == before,
        "pre_resume_record_sha256": before,
        "post_resume_record_sha256": hashes,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if verification_path.exists():
        if verification_path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError("existing resume verification differs")
    else:
        analysis.mkdir(parents=True, exist_ok=True)
        verification_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
