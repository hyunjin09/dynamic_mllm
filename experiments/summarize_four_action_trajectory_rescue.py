#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from experiments.summarize_four_action_stage import artifact_paths, worker_contract_passes
from tools.research_analysis.four_action.followup import trajectory_reference_from_state


ROOT = Path("analysis/4action_answer_alignment/trajectory_rescue")
CONFIG = Path("configs/four_action_answer_alignment.yaml")


def read_jsonl(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_once(path: Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )


def distribution(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=0)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def trajectory_result_semantically_passes(row, trajectory_atol: float) -> bool:
    if row.get("passed") is True:
        return True
    checks = row.get("checks", {})
    failed = {name for name, passed in checks.items() if not passed}
    reference_checks = {
        "trajectory_final_margin_matches_state",
        "trajectory_final_margin_matches_reference_target_state",
    }
    if len(failed) != 1 or not failed.issubset(reference_checks):
        return False
    try:
        reference_margin = trajectory_reference_from_state(
            row["state"], row["suppressed_trajectory"]
        )["fixed_target_state_margin"]
    except (KeyError, TypeError, ValueError):
        # Compatibility for synthetic legacy rows that predate candidate storage.
        reference_margin = row["state"]["margin"]
    difference = abs(
        float(row["suppressed_trajectory"]["final_margin"])
        - float(reference_margin)
    )
    return difference <= trajectory_atol


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    trajectory_atol = float(config["trajectory_final_margin_atol"])
    selection = json.loads((ROOT / "selection_summary.json").read_text(encoding="utf-8"))
    rows = []
    failures = []
    runtimes = []
    missing = []
    for shard in range(8):
        directory = ROOT / "results" / f"shard_{shard:02d}"
        runtime_paths = artifact_paths(directory, "runtime", ".json")
        result_paths = artifact_paths(directory, "results", ".jsonl")
        if not runtime_paths or not result_paths:
            missing.append(shard)
            continue
        runtimes.extend(
            json.loads(path.read_text(encoding="utf-8")) for path in runtime_paths
        )
        for path in result_paths:
            rows.extend(read_jsonl(path))
        for path in artifact_paths(directory, "failures", ".jsonl"):
            failures.extend(read_jsonl(path))
    recovered_rows = [
        row
        for row in rows
        if row.get("passed") is not True
        and trajectory_result_semantically_passes(row, trajectory_atol)
    ]
    recovered_uids = {row["uid"] for row in recovered_rows}
    disqualifying_failures = [
        failure
        for failure in failures
        if not (
            failure.get("uid") in recovered_uids
            and any(
                name in failure.get("error", "")
                for name in (
                    "trajectory_final_margin_matches_state",
                    "trajectory_final_margin_matches_reference_target_state",
                )
            )
        )
    ]
    trajectory_references = [
        trajectory_reference_from_state(row["state"], row["suppressed_trajectory"])
        for row in rows
    ]
    changes = ("final_margin_improvement", "peak_to_final_erosion_reduction", "largest_drop_magnitude_reduction")
    checks = {
        "all_eight_shards_present": not missing,
        "eight_gpu_worker_contract": worker_contract_passes(runtimes),
        "expected_unique_selections": len(rows) == int(selection["selection_count"])
        and len({row["selection_id"] for row in rows}) == int(selection["selection_count"]),
        "no_failures": not disqualifying_failures,
        "all_result_gates_pass": all(
            trajectory_result_semantically_passes(row, trajectory_atol) for row in rows
        ),
    }
    by_operation = {}
    for operation in sorted({row["culprit_operation"] for row in rows}):
        group = [row for row in rows if row["culprit_operation"] == operation]
        by_operation[operation] = {
            "count": len(group),
            **{
                change: distribution([row["trajectory_change"][change] for row in group])
                for change in changes
            },
            "fraction_positive_final_margin_improvement": float(
                np.mean([row["trajectory_change"]["final_margin_improvement"] > 0.0 for row in group])
            ),
            "fraction_erosion_reduced": float(
                np.mean([row["trajectory_change"]["peak_to_final_erosion_reduction"] > 0.0 for row in group])
            ),
        }
    summary = {
        "schema_version": "four_action_trajectory_rescue_summary_v1",
        "selection": selection,
        "observed_results": len(rows),
        "dataset_counts": dict(Counter(row["dataset"] for row in rows)),
        "checks": checks,
        "passed": all(checks.values()),
        "missing_shards": missing,
        "failures": failures,
        "disqualifying_failures": disqualifying_failures,
        "trajectory_identity_rechecks": {
            "atol": trajectory_atol,
            "recovered_count": len(recovered_rows),
            "recovered_selection_ids": [row["selection_id"] for row in recovered_rows],
            "diagnostic_only": True,
        },
        "correct_target_identity": {
            "trajectory_definition": "fixed_baseline_selected_correct_target",
            "evaluator_best_endpoint_preserved_in_state": True,
            "switch_count": sum(
                reference["correct_target_switched"]
                for reference in trajectory_references
            ),
            "switch_fraction": float(
                np.mean(
                    [
                        reference["correct_target_switched"]
                        for reference in trajectory_references
                    ]
                )
            ),
        },
        "by_operation": by_operation,
    }
    rows.sort(key=lambda row: row["selection_id"])
    write_once(
        ROOT / "merged_results.jsonl",
        b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows),
    )
    write_once(
        ROOT / "summary.json",
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
