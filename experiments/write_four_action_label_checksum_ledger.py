#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the conversion artifact checksum ledger.")
    parser.add_argument(
        "--conversion-root",
        type=Path,
        default=Path("datasets/mcts_labels_4action/conversion_v1"),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("datasets/mcts_labels_4action/source_inventory_v1"),
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=Path("analysis/4action_label_conversion"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/4action_label_conversion/checksum_ledger_v1.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    full_audit = json.loads(
        (args.analysis_root / "full_integrity_audit_v1.json").read_text()
    )
    if not full_audit.get("passed"):
        raise RuntimeError("full integrity audit must pass before ledger construction")

    artifacts = []
    candidates = [
        *args.source_root.glob("*"),
        args.conversion_root / "pilot" / "pilot_manifest_v1.jsonl",
        args.conversion_root / "pilot" / "pilot_selection_summary_v1.json",
        args.conversion_root / "full" / "execution_contract_v1.json",
        *(args.conversion_root / "views").glob("*"),
        *args.analysis_root.glob("*.md"),
        *args.analysis_root.glob("*.json"),
        *args.analysis_root.glob("*.csv"),
        *(args.analysis_root / "final").glob("*"),
    ]
    seen = set()
    for path in candidates:
        path = Path(path)
        if not path.is_file() or path.suffix == ".sha256" or path.resolve() == args.output.resolve():
            continue
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        artifacts.append(
            {
                "path": resolved,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    artifacts.sort(key=lambda row: row["path"])

    record_sidecars = sorted(
        (args.conversion_root / "full" / "records").glob("*.json.sha256")
    )
    record_entries = []
    for sidecar in record_sidecars:
        fields = sidecar.read_text().split()
        if len(fields) < 2:
            raise ValueError(f"invalid record checksum sidecar: {sidecar}")
        record_entries.append(f"{fields[1]} {fields[0]}")
    record_merkle = hashlib.sha256(("\n".join(record_entries) + "\n").encode()).hexdigest()
    report = {
        "schema_version": "four_action_label_checksum_ledger_v1",
        "full_integrity_audit_sha256": file_sha256(
            args.analysis_root / "full_integrity_audit_v1.json"
        ),
        "full_record_count": len(record_entries),
        "full_record_sidecar_manifest_sha256": record_merkle,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{file_sha256(args.output)}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
