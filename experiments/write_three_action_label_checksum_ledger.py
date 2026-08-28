#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from tools.research_analysis.four_action.three_action_jobs import file_sha256


def build_checksum_ledger(
    candidates: Iterable[Path],
    *,
    record_sidecars: Iterable[Path],
    full_audit_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    artifacts = []
    seen = set()
    for candidate in candidates:
        path = Path(candidate)
        if (
            not path.is_file()
            or path.suffix == ".sha256"
            or path.resolve() == output_path.resolve()
        ):
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

    record_entries = []
    for sidecar in sorted(Path(path) for path in record_sidecars):
        fields = sidecar.read_text(encoding="utf-8").split()
        if len(fields) < 2:
            raise ValueError(f"invalid record checksum sidecar: {sidecar}")
        record_entries.append(f"{fields[1]} {fields[0]}")
    record_manifest = "\n".join(record_entries) + ("\n" if record_entries else "")
    return {
        "schema_version": "three_action_answer_aligned_checksum_ledger_v1",
        "full_integrity_audit_sha256": file_sha256(full_audit_path),
        "full_record_count": len(record_entries),
        "full_record_sidecar_manifest_sha256": sha256(
            record_manifest.encode("utf-8")
        ).hexdigest(),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the three-action conversion checksum ledger.")
    parser.add_argument("--config", type=Path, default=Path("configs/three_action_label_conversion.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/three_action_answer_aligned_label_conversion/checksum_ledger_v1.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = Path(config["output_root"])
    analysis_root = Path(config["analysis_root"])
    full_audit_path = analysis_root / "full_integrity_audit_v1.json"
    full_audit = json.loads(full_audit_path.read_text(encoding="utf-8"))
    finalization = json.loads(
        (output_root / "final" / "finalization_summary_v1.json").read_text(encoding="utf-8")
    )
    if not full_audit.get("passed") or not finalization.get("passed"):
        raise RuntimeError("full audit and finalization must pass before ledger construction")

    required = [
        output_root / "calibration" / "noise_calibration_v1.json",
        output_root / "pilot" / "pilot_manifest_v1.jsonl",
        output_root / "pilot" / "pilot_selection_summary_v1.json",
        output_root / "full" / "execution_contract_v1.json",
        output_root / "final" / "finalization_summary_v1.json",
        full_audit_path,
        analysis_root / "aggregate_statistics_v1.json",
        analysis_root / "screening_positions_v1.csv",
        analysis_root / "decomposition_actions_v1.csv",
        analysis_root / "screening_classification_by_layer.png",
        analysis_root / "three_action_answer_aligned_label_conversion_report.md",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("required final artifacts are missing: " + ", ".join(missing))
    candidates = [
        Path(config["source_manifest"]),
        Path(config["source_summary"]),
        *required,
        *(output_root / "final").glob("*"),
        *analysis_root.glob("*.md"),
        *analysis_root.glob("*.json"),
        *analysis_root.glob("*.csv"),
        *analysis_root.glob("*.png"),
    ]
    report = build_checksum_ledger(
        candidates,
        record_sidecars=(output_root / "full" / "records").glob("*.json.sha256"),
        full_audit_path=full_audit_path,
        output_path=args.output,
    )
    if int(report["full_record_count"]) != int(full_audit["completed_samples"]):
        raise RuntimeError("record checksum count does not match the passing full audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{file_sha256(args.output)}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
