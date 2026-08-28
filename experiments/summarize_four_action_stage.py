#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tools.research_analysis.four_action.targets import answer_targets_are_scorable


MODES = (
    "preflight",
    "smoke",
    "pilot",
    "primary",
    "control_no_correction",
    "control_vision_required",
)


def artifact_paths(directory: Path, stem: str, suffix: str) -> list[Path]:
    return sorted(
        path
        for path in directory.glob(f"{stem}*{suffix}")
        if path.name == f"{stem}{suffix}"
        or path.name.startswith(f"{stem}_replica_")
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_once(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )


def expected_count(
    mode: str,
    cohort_summary: dict[str, Any],
    eligibility_summary: dict[str, Any] | None = None,
    target_unscorable_count: int = 0,
) -> int:
    if mode in {"preflight", "smoke"}:
        return len(cohort_summary["smoke_ids"])
    if mode == "pilot":
        return len(cohort_summary["pilot_ids"])
    if mode == "primary":
        return (
            int(eligibility_summary["eligible_counts"]["primary_a_plus"])
            if eligibility_summary is not None
            else int(cohort_summary["primary_rows"])
        )
    key = {
        "control_no_correction": "control_no_correction_found",
        "control_vision_required": "control_full_correct_all_off_wrong",
    }[mode]
    if eligibility_summary is not None:
        return int(eligibility_summary["eligible_counts"][key]) - target_unscorable_count
    return (
        sum(int(dataset[key]) for dataset in cohort_summary["taxonomy"].values())
        - target_unscorable_count
    )


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(array.max()),
    }


def drift_distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(array.max()),
    }


def sample_gate_semantically_passes(
    row: dict[str, Any], trajectory_atol: float | None = None
) -> bool:
    gate = row.get("sample_gate", {})
    if gate.get("passed") is True:
        return True
    checks = gate.get("checks", {})
    failed = {name for name, passed in checks.items() if not passed}
    if failed == {"baseline_cached_ids_match"}:
        return True
    if failed == {"trajectory_final_margin_identity"} and trajectory_atol is not None:
        difference = row.get("unified_full_answer_trajectory", {}).get(
            "final_margin_vs_factorial_baseline_abs_diff"
        )
        return difference is not None and float(difference) <= trajectory_atol
    return False


def worker_contract_passes(runtimes: list[dict[str, Any]]) -> bool:
    if not runtimes:
        return False
    replicas = [int(row.get("replicas_per_gpu", 1)) for row in runtimes]
    active_replicas = max(replicas)
    active = [
        row
        for row in runtimes
        if int(row.get("replicas_per_gpu", 1)) == active_replicas
    ]
    observed = {
        (
            int(row.get("gpu_index", row["rank"])),
            int(row.get("replica_index", 0)),
        )
        for row in active
    }
    expected = {
        (gpu_index, replica_index)
        for gpu_index in range(8)
        for replica_index in range(active_replicas)
    }
    return observed == expected and all(
        int(row["world_size"]) == 8 * active_replicas for row in active
    )


def historical_anchor_ids_match(row: dict[str, Any]) -> bool | None:
    routes = row.get("binary_routes")
    if routes is None:
        return None
    diagnostics = row.get("sample_gate", {}).get("diagnostics", {})
    if "historical_full_anchor_generated_ids_match" in diagnostics:
        return bool(diagnostics["historical_full_anchor_generated_ids_match"])
    return (
        row["native_full_external"]["state"]["generated_ids"]
        == routes["full_anchor"]["generated_ids"]
    )


def summarize(
    mode: str, config: dict[str, Any], output_tag: str = ""
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mode_directory = mode if not output_tag else f"{mode}__{output_tag}"
    root = Path(config["output_root"]) / mode_directory
    rows: list[dict[str, Any]] = []
    runtimes = []
    failure_rows = []
    missing_shards = []
    for shard in range(8):
        shard_dir = root / f"shard_{shard:02d}"
        result_paths = artifact_paths(shard_dir, "results", ".jsonl")
        runtime_paths = artifact_paths(shard_dir, "runtime", ".json")
        if not result_paths or not runtime_paths:
            missing_shards.append(shard)
            continue
        for result_path in result_paths:
            rows.extend(read_jsonl(result_path))
        for runtime_path in runtime_paths:
            runtimes.append(json.loads(runtime_path.read_text(encoding="utf-8")))
        for failure_path in artifact_paths(shard_dir, "failures", ".jsonl"):
            failure_rows.extend(read_jsonl(failure_path))
    cohort_summary = json.loads(Path(config["cohort_summary"]).read_text(encoding="utf-8"))
    eligibility_summary = None
    eligible_ids = None
    raw_excluded_rows = []
    target_unscorable_rows = []
    if mode not in {"preflight", "smoke", "pilot"}:
        eligibility_root = Path(config["eligibility_root"])
        eligibility_summary = json.loads((eligibility_root / "summary.json").read_text())
        if not eligibility_summary.get("passed"):
            raise RuntimeError("unified-FULL eligibility freeze did not pass")
        eligibility_rows = read_jsonl(eligibility_root / "merged_results.jsonl")
        frozen_eligible_ids = {row["uid"] for row in eligibility_rows if row["eligible"]}
        if mode == "control_no_correction":
            manifest_rows = read_jsonl(Path(config["cohort_manifest"]))
            target_unscorable_rows = [
                {
                    "uid": row["uid"],
                    "dataset": row["dataset"],
                    "cohort": row["cohort"],
                    "reason": "no evaluator-valid correct answer target",
                }
                for row in manifest_rows
                if row["uid"] in frozen_eligible_ids
                and row["cohort"] == "control_no_correction_found"
                and not answer_targets_are_scorable(row)
            ]
        target_unscorable_ids = {row["uid"] for row in target_unscorable_rows}
        eligible_ids = frozen_eligible_ids - target_unscorable_ids
        raw_excluded_rows = [row for row in rows if row["uid"] not in frozen_eligible_ids]
        rows = [row for row in rows if row["uid"] in eligible_ids]
    expected = expected_count(
        mode,
        cohort_summary,
        eligibility_summary,
        target_unscorable_count=len(target_unscorable_rows),
    )
    ids = [row["uid"] for row in rows]
    layer_count = len(config["preflight_layer_grid"] if mode == "preflight" else config["layer_grid"])
    trajectory_atol = float(config["trajectory_final_margin_atol"])
    semantically_valid_ids = {
        row["uid"]
        for row in rows
        if sample_gate_semantically_passes(row, trajectory_atol)
    }
    disqualifying_failures = [
        failure for failure in failure_rows
        if not (
            (eligible_ids is not None and failure.get("uid") not in eligible_ids)
            or (
                failure.get("uid") in semantically_valid_ids
                and failure.get("error")
                in {
                    "sample gate failed: ['baseline_cached_ids_match']",
                    "sample gate failed: ['trajectory_final_margin_identity']",
                }
            )
        )
    ]
    anchor_disagreements = [
        {
            "uid": row["uid"],
            "dataset": row["dataset"],
            "current_native_generated_ids": row["native_full_external"]["state"]["generated_ids"],
            "current_native_generated_answer": row["native_full_external"]["state"]["generated_answer"],
            "historical_generated_ids": row["binary_routes"]["full_anchor"]["generated_ids"],
            "historical_generated_answer": row["binary_routes"]["full_anchor"]["prediction"],
            "current_correct": row["native_full_external"]["state"]["correct"],
            "historical_correct": row["binary_routes"]["full_anchor"]["correct"],
        }
        for row in rows
        if historical_anchor_ids_match(row) is False
    ]
    checks = {
        "all_eight_shards_present": not missing_shards,
        "eight_worker_contract": worker_contract_passes(runtimes),
        "expected_unique_records": len(rows) == expected and len(set(ids)) == expected,
        "no_disqualifying_failure_records": not disqualifying_failures,
        "sample_semantic_gates_pass": all(
            sample_gate_semantically_passes(row, trajectory_atol) for row in rows
        ),
        "expected_layer_count": all(len(row["layers"]) == layer_count for row in rows),
        "preflight_gates_pass": mode != "preflight"
        or all(row.get("preflight_gate", {}).get("passed") is True for row in rows),
    }
    if mode in {"preflight", "smoke", "pilot"}:
        checks["unified_full_native_semantic_parity"] = all(
            all(
                row["native_full_external"]["diagnostic"][name]
                for name in (
                    "generated_ids_match",
                    "generated_answer_match",
                    "evaluator_score_match",
                    "correctness_match",
                )
            )
            for row in rows
        )
        checks["unified_ignore_binary_semantic_parity"] = all(
            all(
                all(
                    layer["old_binary_ignore_external"][name]
                    for name in (
                        "generated_ids_match",
                        "generated_answer_match",
                        "evaluator_score_match",
                        "correctness_match",
                    )
                )
                for layer in row["layers"]
            )
            for row in rows
        )
    elapsed = [float(row["elapsed_seconds"]) for row in rows]
    per_dataset = Counter(row["dataset"] for row in rows)
    local_rescues = Counter()
    for row in rows:
        for layer in row["layers"]:
            for action in ("IGNORE", "READ_ONLY", "WRITE_ONLY"):
                if layer["states"][action]["correct"]:
                    local_rescues[(row["dataset"], action)] += 1
    timing = quantiles(elapsed) if elapsed else None
    margin_signed_drift = [
        float(row["native_full_external"]["diagnostic"]["signed_drift"]["margin"])
        for row in rows
    ]
    native_unified_drift = None
    if margin_signed_drift:
        native_unified_drift = {
            "definition": "unified_materialized_full_minus_native_maskless_full",
            "margin_signed": drift_distribution(margin_signed_drift),
            "margin_absolute": drift_distribution([abs(value) for value in margin_signed_drift]),
            "diagnostic_only_not_an_effect_threshold": True,
        }
    estimate = None
    if timing is not None:
        mean = timing["mean"]
        validation_overhead = mode in {"preflight", "smoke", "pilot"}
        estimate = {
            "primary_gpu_hours": mean * int(cohort_summary["primary_rows"]) / 3600.0,
            "primary_wall_hours_at_eight_workers": mean * int(cohort_summary["primary_rows"]) / (3600.0 * 8),
            "all_controls_gpu_hours": mean
            * expected_count("control_no_correction", cohort_summary)
            / 3600.0
            + mean * expected_count("control_vision_required", cohort_summary) / 3600.0,
            "basis": (
                "mean observed validation seconds per sample; includes old-binary "
                "single-OFF semantic checks omitted from production, so primary/control "
                "estimates are conservative upper estimates"
                if validation_overhead
                else "mean observed production seconds per sample at the same scope"
            ),
            "validation_overhead_included": validation_overhead,
        }
    summary = {
        "schema_version": "four_action_stage_summary_v1",
        "mode": mode,
        "output_tag": output_tag,
        "expected_records": expected,
        "observed_records": len(rows),
        "dataset_counts": dict(per_dataset),
        "layer_count": layer_count,
        "checks": checks,
        "passed": all(checks.values()),
        "missing_shards": missing_shards,
        "worker_layouts": [
            {
                "rank": runtime["rank"],
                "world_size": runtime["world_size"],
                "gpu_index": runtime.get("gpu_index", runtime["rank"]),
                "replica_index": runtime.get("replica_index", 0),
                "replicas_per_gpu": runtime.get("replicas_per_gpu", 1),
            }
            for runtime in runtimes
        ],
        "failure_rows": failure_rows,
        "disqualifying_failure_rows": disqualifying_failures,
        "raw_rows_excluded_by_unified_full_eligibility": [
            {"uid": row["uid"], "dataset": row["dataset"], "cohort": row["cohort"]}
            for row in raw_excluded_rows
        ],
        "target_unscorable_exclusions": target_unscorable_rows,
        "target_unscorable_exclusion_rule": (
            "exclude only when no frozen reference can meet the evaluator correctness "
            "threshold, so correct-vs-FULL-wrong margin is undefined"
        ),
        "historical_full_anchor_token_identity": {
            "comparison_count": sum(historical_anchor_ids_match(row) is not None for row in rows),
            "match_count": sum(historical_anchor_ids_match(row) is True for row in rows),
            "disagreements": anchor_disagreements,
            "diagnostic_only": True,
            "gate_rule": (
                "current native FULL must preserve cohort correctness and match unified FULL; "
                "exact token identity to the transferred historical anchor is provenance only"
            ),
        },
        "timing_seconds_per_sample": timing,
        "compute_estimate": estimate,
        "native_unified_full_drift": native_unified_drift,
        "local_rescue_cells_diagnostic_only": {
            f"{dataset}/{action}": count
            for (dataset, action), count in sorted(local_rescues.items())
        },
    }
    return sorted(rows, key=lambda row: (row["dataset"], row["uid"])), summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and gate one eight-shard four-action stage.")
    parser.add_argument("--config", type=Path, default=Path("configs/four_action_answer_alignment.yaml"))
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows, summary = summarize(args.mode, config, args.output_tag)
    mode_directory = args.mode if not args.output_tag else f"{args.mode}__{args.output_tag}"
    root = Path(config["output_root"]) / mode_directory
    write_once(root / "merged_results.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    write_once(root / "stage_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
