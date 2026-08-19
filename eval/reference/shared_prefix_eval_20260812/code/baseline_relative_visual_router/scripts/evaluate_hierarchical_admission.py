#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from analysis_outputs.dense_prefill_hierarchical_gate import (  # noqa: E402
    FeatureScaler,
    load_feature_parts,
    transform_features,
)
from baseline_relative_visual_router.admission import summarize_admission  # noqa: E402
from baseline_relative_visual_router.utility import conservative_utility_score  # noqa: E402
from train_actual_policy_admission import (  # noqa: E402
    bootstrap_gate,
    member_probabilities,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harm-experiment", type=Path, required=True)
    parser.add_argument("--utility-experiment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def decision_summary(rows: list[dict[str, Any]], admission: np.ndarray) -> dict[str, Any]:
    score = np.where(admission, 0.0, 1.0)
    return summarize_admission(rows, score, threshold=0.5)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(max(1, min(2, args.cpu_threads)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    harm_summary = json.loads((args.harm_experiment / "summary.json").read_text())
    utility_summary = json.loads((args.utility_experiment / "summary.json").read_text())
    if harm_summary["split_uids"] != utility_summary["split_uids"]:
        raise RuntimeError("harm and utility experiments use different splits")
    harm_checkpoint = torch.load(
        args.harm_experiment / "admission_gate.pt", map_location="cpu", weights_only=False
    )
    utility_checkpoint = torch.load(
        args.utility_experiment / "utility_admission_gate.pt",
        map_location="cpu",
        weights_only=False,
    )
    tensors, metadata = load_feature_parts(Path(harm_summary["feature_dir"]))
    policy_rows = read_jsonl(Path(harm_summary["policy_rows"]))
    by_uid = {str(row["uid"]): row for row in policy_rows}
    rows = [by_uid[str(meta["uid"])] for meta in metadata]
    uid_to_index = {str(meta["uid"]): index for index, meta in enumerate(metadata)}
    features = transform_features(
        tensors, FeatureScaler.from_state_dict(harm_checkpoint["scaler"])
    )
    device = torch.device(args.device)

    harm_candidate = harm_summary["selected_candidate"]
    harm_architecture = str(harm_candidate["architecture"])
    harm_members = member_probabilities(
        features,
        harm_checkpoint["models"][harm_architecture],
        harm_architecture,
        device,
    )
    harm_score = harm_members.mean(axis=0) + float(
        harm_candidate["uncertainty_beta"]
    ) * harm_members.std(axis=0)
    harm_threshold = float(harm_candidate["calibration"]["threshold"])

    utility_candidate = utility_summary["selected_performance_candidate"]
    harm_architecture = str(utility_candidate["harm_architecture"])
    rescue_architecture = str(utility_candidate["rescue_architecture"])
    utility_harm_members = member_probabilities(
        features,
        utility_checkpoint["states"]["harm"][harm_architecture],
        harm_architecture,
        device,
    )
    utility_rescue_members = member_probabilities(
        features,
        utility_checkpoint["states"]["rescue"][rescue_architecture],
        rescue_architecture,
        device,
    )
    utility_score = conservative_utility_score(
        utility_harm_members,
        utility_rescue_members,
        uncertainty_beta=float(utility_candidate["uncertainty_beta"]),
        rescue_weight=float(utility_candidate["rescue_weight"]),
    )
    utility_threshold = float(
        utility_candidate["calibration_performance"]["threshold"]
    )

    output: dict[str, Any] = {
        "schema_version": "hierarchical_actual_policy_admission_v1",
        "policy": harm_summary["policy"],
        "feature_contract": harm_summary["feature_contract"],
        "decision": "safe_efficiency_admission OR high_confidence_rescue_override",
        "thresholds": {
            "safe_harm": harm_threshold,
            "rescue_utility": utility_threshold,
        },
        "splits": {},
    }
    prediction_rows = []
    for split in ("calibration", "test"):
        indices = np.asarray(
            [uid_to_index[uid] for uid in harm_summary["split_uids"][split]],
            dtype=np.int64,
        )
        split_rows = [rows[int(index)] for index in indices]
        safe = harm_score[indices] <= harm_threshold
        rescue = utility_score[indices] <= utility_threshold
        combined = safe | rescue
        summaries = {
            "safe_efficiency": decision_summary(split_rows, safe),
            "rescue_override": decision_summary(split_rows, rescue),
            "combined": decision_summary(split_rows, combined),
        }
        summaries["combined"].update(
            bootstrap_gate(
                split_rows,
                np.where(combined, 0.0, 1.0),
                0.5,
                args.bootstrap_repetitions,
                20260812 + (0 if split == "calibration" else 1),
            )
        )
        benchmarks = np.asarray(
            [str(metadata[int(index)]["benchmark"]) for index in indices], dtype=object
        )
        summaries["combined_by_benchmark"] = {}
        for benchmark in sorted(set(benchmarks)):
            keep = np.flatnonzero(benchmarks == benchmark)
            summaries["combined_by_benchmark"][benchmark] = decision_summary(
                [split_rows[int(index)] for index in keep], combined[keep]
            )
        output["splits"][split] = summaries
        if split == "test":
            for local_index, global_index in enumerate(indices):
                row = split_rows[local_index]
                prediction_rows.append(
                    {
                        "uid": str(metadata[int(global_index)]["uid"]),
                        "benchmark": str(metadata[int(global_index)]["benchmark"]),
                        "safe_efficiency_admission": bool(safe[local_index]),
                        "rescue_override": bool(rescue[local_index]),
                        "combined_admission": bool(combined[local_index]),
                        "harm_score": float(harm_score[global_index]),
                        "utility_score": float(utility_score[global_index]),
                        "baseline_correct": bool(row["baseline_correct"]),
                        "router_correct": bool(row["router_correct"]),
                        "selected_num_visual_on_layers": int(
                            row["selected_num_visual_on_layers"]
                        ),
                    }
                )
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "locked_test_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(output["splits"], indent=2), flush=True)


if __name__ == "__main__":
    main()
