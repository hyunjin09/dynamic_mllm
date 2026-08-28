#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterator

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from tools.research_analysis.four_action.sequential_label_analysis import (
    ACTIONS,
    aggregate_records,
)
from tools.research_analysis.four_action.sequential_label_jobs import file_sha256


def iter_records(root: Path) -> Iterator[dict[str, Any]]:
    for path in sorted(root.glob("*.json")):
        yield json.loads(path.read_text(encoding="utf-8"))


def write_or_verify(path: Path, content: str, *, resume: bool) -> None:
    if path.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {path}")
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"existing analysis differs from recomputation: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{file_sha256(path)}  {path.name}\n", encoding="utf-8"
    )


def _plots(aggregate: dict[str, Any], plot_root: Path, *, resume: bool) -> None:
    plot_root.mkdir(parents=True, exist_ok=True)
    datasets = list(aggregate["by_dataset"])
    fractions = [
        [
            aggregate["by_dataset"][dataset]["w2c"][
                "off_position_final_action_fractions"
            ][action]
            or 0.0
            for dataset in datasets
        ]
        for action in ACTIONS
    ]
    target = plot_root / "w2c_final_action_fraction_by_dataset.png"
    if not target.exists() or not resume:
        fig, axis = plt.subplots(figsize=(10, 5))
        bottom = [0.0] * len(datasets)
        for action, values in zip(ACTIONS, fractions, strict=True):
            axis.bar(datasets, values, bottom=bottom, label=action)
            bottom = [left + right for left, right in zip(bottom, values, strict=True)]
        axis.set_ylabel("Fraction across final W→C branches at source-OFF positions")
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=25)
        axis.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(target, dpi=160, metadata={"Date": None})
        plt.close(fig)

    layers = aggregate["combined"]["w2c"]["off_position_actions_by_layer"]
    target = plot_root / "w2c_final_action_fraction_by_layer.png"
    if not target.exists() or not resume:
        x = list(range(len(layers)))
        fig, axis = plt.subplots(figsize=(12, 5))
        bottom = [0.0] * len(layers)
        for action in ACTIONS:
            values = []
            for row in layers:
                total = sum(int(row[value]) for value in ACTIONS)
                values.append(int(row[action]) / total if total else 0.0)
            axis.bar(x, values, bottom=bottom, label=action, width=0.9)
            bottom = [left + right for left, right in zip(bottom, values, strict=True)]
        axis.set_xlabel("Layer")
        axis.set_ylabel("Fraction across final W→C branches")
        axis.set_ylim(0, 1)
        axis.legend(ncol=4)
        fig.tight_layout()
        fig.savefig(target, dpi=160, metadata={"Date": None})
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze exact sequential four-action labels.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sequential_four_action_label_conversion.yaml"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    full_root = Path(config["output_root"]) / "full"
    audit = json.loads((full_root / "completion_audit_v1.json").read_text(encoding="utf-8"))
    if not audit.get("passed"):
        raise RuntimeError("full completion audit did not pass")
    aggregate = aggregate_records(
        iter_records(full_root / "records"), layer_count=int(config["layer_count"])
    )
    aggregate["source_inventory"] = json.loads(
        Path(config["source_summary"]).read_text(encoding="utf-8")
    )
    aggregate["completion_audit"] = audit
    estimate_path = Path(config["analysis_root"]) / "full_compute_estimate_v1.json"
    aggregate["compute_estimate"] = (
        json.loads(estimate_path.read_text(encoding="utf-8"))
        if estimate_path.exists()
        else None
    )
    output = Path(config["analysis_root"]) / "aggregate_statistics_v1.json"
    write_or_verify(
        output,
        json.dumps(aggregate, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        resume=args.resume,
    )
    _plots(aggregate, Path(config["analysis_root"]) / "plots", resume=args.resume)
    print(json.dumps(aggregate["combined"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
