#!/usr/bin/env python3
"""Create deterministic tables from completed Pareto checkpoint evaluations."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


TRAJECTORY_METRICS = (
    "pareto_valid_hit_at_1",
    "original_valid_hit_at_1",
    "nearest_valid_hamming",
    "average_predicted_visual_on",
    "fraction_top1_all_on",
    "fraction_top1_all_off",
    "unique_top1_masks",
    "top1_mask_entropy_nats",
)


def first_best_epoch(rows: list[dict[str, Any]], key, *, maximize: bool) -> int:
    values = [(int(row["epoch"]), float(key(row))) for row in rows]
    optimum = (max if maximize else min)(value for _, value in values)
    return min(epoch for epoch, value in values if value == optimum)


def epoch_selection(raw: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "best_train_pareto_hit_epoch": first_best_epoch(
            raw["epochs"], lambda row: row["train"]["overall"]["pareto_valid_hit_at_1"], maximize=True
        ),
        "best_validation_pareto_hit_epoch": first_best_epoch(
            history, lambda row: row["validation"]["overall"]["pareto_valid_hit_at_1"], maximize=True
        ),
        "lowest_online_train_loss_epoch": first_best_epoch(
            history, lambda row: row["train"]["loss"], maximize=False
        ),
        "lowest_validation_loss_epoch": first_best_epoch(
            history, lambda row: row["validation"]["overall"]["objective_loss"], maximize=False
        ),
        "final_epoch": 10,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trajectory_rows(objective: str, raw: dict[str, Any], history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for evaluated, logged in zip(raw["epochs"], history, strict=True):
        train = evaluated["train"]["overall"]
        validation = logged["validation"]["overall"]
        row: dict[str, Any] = {
            "objective": objective,
            "epoch": evaluated["epoch"],
            "online_train_loss": logged["train"]["loss"],
            "checkpoint_train_objective_loss": train["objective_loss"],
            "logged_validation_objective_loss": validation["objective_loss"],
        }
        for metric in TRAJECTORY_METRICS:
            row[f"train_{metric}"] = train[metric]
            row[f"validation_{metric}"] = validation[metric]
            row[f"train_minus_validation_{metric}"] = train[metric] - validation[metric]
        rows.append(row)
    return rows


def stratum_rows(
    objective: str,
    raw: dict[str, Any],
    history: list[dict[str, Any]],
    section: str,
) -> list[dict[str, Any]]:
    rows = []
    for evaluated, logged in zip(raw["epochs"], history, strict=True):
        for stratum, train in evaluated["train"][section].items():
            validation = logged["validation"][section][stratum]
            for split, metrics in (("train", train), ("validation", validation)):
                rows.append(
                    {
                        "objective": objective,
                        "epoch": evaluated["epoch"],
                        "split": split,
                        "stratum": stratum,
                        "examples": metrics["examples"],
                        "pareto_valid_hit_at_1": metrics["pareto_valid_hit_at_1"],
                        "original_valid_hit_at_1": metrics["original_valid_hit_at_1"],
                        "nearest_valid_hamming": metrics["nearest_valid_hamming"],
                        "average_predicted_visual_on": metrics["average_predicted_visual_on"],
                        "fraction_top1_all_on": metrics["fraction_top1_all_on"],
                    }
                )
    return rows


def probability_rows(objective: str, raw: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for evaluated in raw["epochs"]:
        for split in ("train", "validation"):
            if split not in evaluated:
                continue
            rows.append(
                {
                    "objective": objective,
                    "epoch": evaluated["epoch"],
                    "split": split,
                    **evaluated[split]["probability_diagnostics"],
                }
            )
    return rows


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bce", type=Path, required=True)
    parser.add_argument("--nll", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = {"duplicated_bce": args.bce, "exact_set_nll": args.nll}
    all_trajectories: dict[str, list[dict[str, Any]]] = {}
    all_multiplicity = []
    all_groups = []
    all_probability = []
    selections = {}
    for objective, path in inputs.items():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not raw.get("passed") or raw["objective"] != objective or len(raw["epochs"]) != 10:
            raise RuntimeError(f"incomplete or mismatched checkpoint evaluation: {path}")
        history_path = Path(raw["training_dir"]) / "history.json"
        history = json.loads(history_path.read_text(encoding="utf-8"))
        trajectories = trajectory_rows(objective, raw, history)
        all_trajectories[objective] = trajectories
        all_multiplicity.extend(
            stratum_rows(objective, raw, history, "by_pareto_multiplicity")
        )
        all_groups.extend(
            stratum_rows(objective, raw, history, "by_supervision_group")
        )
        all_probability.extend(probability_rows(objective, raw))
        selections[objective] = epoch_selection(raw, history)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for objective, filename in (
        ("duplicated_bce", "bce_epoch_trajectory_v1.csv"),
        ("exact_set_nll", "nll_epoch_trajectory_v1.csv"),
    ):
        path = args.output_dir / filename
        write_csv(path, all_trajectories[objective])
        written.append(path)
    for filename, rows in (
        ("multiplicity_metrics_v1.csv", all_multiplicity),
        ("supervision_group_metrics_v1.csv", all_groups),
        ("probability_metrics_v1.csv", all_probability),
    ):
        path = args.output_dir / filename
        write_csv(path, rows)
        written.append(path)
    selection_path = args.output_dir / "epoch_selection_v1.json"
    selection_path.write_text(json.dumps(selections, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written.append(selection_path)
    manifest_path = args.output_dir / "training_fit_analysis_manifest.json"
    manifest = {
        "schema_version": "binary_pareto_training_fit_analysis_v1",
        "inputs": {str(path): file_sha256(path) for path in inputs.values()},
        "outputs": {str(path): file_sha256(path) for path in written},
        "validation_metric_source": "original frozen A6000 training histories",
        "checkpoint_train_metric_source": "read-only A4000 BF16 reevaluation",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = file_sha256(manifest_path)
    manifest_path.with_suffix(".json.sha256").write_text(
        f"{digest}  {manifest_path.name}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
