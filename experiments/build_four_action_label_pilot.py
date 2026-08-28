#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.research_analysis.four_action.label_jobs import select_conversion_pilot


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_once(path: Path, content: str) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the five-dataset conversion pilot.")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("datasets/mcts_labels_4action/source_inventory_v1/source_manifest_v1.jsonl"),
    )
    parser.add_argument("--total", type=int, default=56)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/mcts_labels_4action/conversion_v1/pilot"),
    )
    args = parser.parse_args()
    rows = read_jsonl(args.source_manifest)
    selected, coverage = select_conversion_pilot(rows, total=args.total)
    manifest = args.output_dir / "pilot_manifest_v1.jsonl"
    manifest_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected)
    digest = write_once(manifest, manifest_text)
    summary = {
        "schema_version": "four_action_label_conversion_pilot_summary_v1",
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": hashlib.sha256(args.source_manifest.read_bytes()).hexdigest(),
        "pilot_manifest_sha256": digest,
        "pilot_count": len(selected),
        "coverage": coverage,
        "dataset_counts": {
            dataset: sum(row["dataset"] == dataset for row in selected)
            for dataset in sorted({row["dataset"] for row in selected})
        },
        "selection_rule": (
            "five-dataset quota; prioritize source W2C ALL-OFF, W2C, C2C, multi-route, "
            "short/long routes, high-route/high-cost samples; fill deterministically by cost"
        ),
    }
    write_once(
        args.output_dir / "pilot_selection_summary_v1.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
