#!/usr/bin/env python3
"""Write the strict P4 audit and checksum-bound terminal-record index."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from label_regeneration.audit import audit_cache


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--record-index", required=True)
    args = parser.parse_args()

    report, index_rows = audit_cache(
        args.manifest,
        args.output_root,
        contract_sha256=args.contract_sha256,
        expected_dataset_counts={"gqa": 4000, "textvqa": 2000, "chartqa": 2000},
    )
    report_path = Path(args.report)
    index_path = Path(args.record_index)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in index_rows),
        encoding="utf-8",
    )
    report["record_index_path"] = str(index_path)
    report["record_index_sha256"] = digest(index_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for path in (report_path, index_path):
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{digest(path)}  {path.name}\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "expected": report["expected_records"],
                "valid": report["valid_terminal_records"],
                "invalid": len(report["invalid_records"]),
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
