from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SEED = 2026080714
FEATURES = (
    "mean_prompt_tokens",
    "prompt_token_difference",
    "mean_answer_tokens",
    "answer_token_difference",
    "mean_program_depth",
    "program_depth_difference",
    "question_type_mismatch_count",
    "pair_match_distance",
    "log_visual_token_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze outcome-blind v4 semantic-control covariate matching."
    )
    parser.add_argument(
        "--manifest",
        default="outputs/v4_discovery/manifest/v4_gqa_discovery_manifest_v1.jsonl",
    )
    parser.add_argument(
        "--output",
        default="outputs/v4_discovery/manifest/v4_semantic_covariate_matching_v1.json",
    )
    return parser.parse_args()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def type_mismatches(first: dict[str, Any], second: dict[str, Any]) -> int:
    left = first.get("question_types") or {}
    right = second.get("question_types") or {}
    return sum(left.get(key) != right.get(key) for key in ("structural", "semantic", "detailed"))


def group_features(group: list[dict[str, Any]]) -> dict[str, float]:
    if len(group) != 2:
        raise ValueError("Every discovery image must have exactly two questions")
    first, second = sorted(group, key=lambda row: int(row["question_index"]))
    prompt = [float(row["expected_prompt_token_length"]) for row in (first, second)]
    answer = [float(row["answer_token_length"]) for row in (first, second)]
    depth = [float(row["semantic_program_depth"]) for row in (first, second)]
    return {
        "mean_prompt_tokens": float(np.mean(prompt)),
        "prompt_token_difference": abs(prompt[0] - prompt[1]),
        "mean_answer_tokens": float(np.mean(answer)),
        "answer_token_difference": abs(answer[0] - answer[1]),
        "mean_program_depth": float(np.mean(depth)),
        "program_depth_difference": abs(depth[0] - depth[1]),
        "question_type_mismatch_count": float(type_mismatches(first, second)),
        "pair_match_distance": float(first["pair_match_distance"]),
        "log_visual_token_count": math.log1p(float(first["expected_visual_token_count"])),
    }


def minimum_cost_assignment(cost: np.ndarray) -> list[int]:
    """Return the minimum-cost column for each row of a square matrix.

    This is the O(n^3) Hungarian shortest-augmenting-path algorithm. Input row
    and column orders are frozen by SHA256 before it is called, which also
    makes exact cost ties deterministic.
    """
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("Assignment cost matrix must be square")
    size = cost.shape[0]
    u = np.zeros(size + 1, dtype=np.float64)
    v = np.zeros(size + 1, dtype=np.float64)
    p = np.zeros(size + 1, dtype=np.int64)
    way = np.zeros(size + 1, dtype=np.int64)
    for row in range(1, size + 1):
        p[0] = row
        minimum = np.full(size + 1, np.inf, dtype=np.float64)
        used = np.zeros(size + 1, dtype=bool)
        column0 = 0
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = np.inf
            column1 = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                current = cost[row0 - 1, column - 1] - u[row0] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(size + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment = [-1] * size
    for column in range(1, size + 1):
        assignment[p[column] - 1] = column - 1
    if any(column < 0 for column in assignment):
        raise RuntimeError("Assignment did not cover every row")
    return assignment


def freeze_matching(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["image_id"])].append(row)
    metadata = {
        image_id: {
            "image_id": image_id,
            "stratum": group[0]["pair_stratum"],
            "features": group_features(group),
        }
        for image_id, group in groups.items()
    }
    treated = [row for row in metadata.values() if row["stratum"] == "different_evidence"]
    controls = [row for row in metadata.values() if row["stratum"] == "matched_comparison"]
    if len(treated) != 60 or len(controls) != 60:
        raise ValueError("Expected 60 images in each semantic stratum")

    matrix = np.asarray(
        [[row["features"][feature] for feature in FEATURES] for row in treated + controls],
        dtype=np.float64,
    )
    center = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=1)
    scale[scale == 0] = 1.0

    def vector(row: dict[str, Any]) -> np.ndarray:
        raw = np.asarray([row["features"][feature] for feature in FEATURES], dtype=np.float64)
        return (raw - center) / scale

    treated = sorted(treated, key=lambda row: stable_hash(SEED, "target", row["image_id"]))
    controls = sorted(controls, key=lambda row: stable_hash(SEED, "control", row["image_id"]))
    cost = np.asarray(
        [
            [float(np.linalg.norm(vector(target) - vector(control))) for control in controls]
            for target in treated
        ],
        dtype=np.float64,
    )
    assignment = minimum_cost_assignment(cost)
    matches = []
    for row_index, target in enumerate(treated):
        donor = controls[assignment[row_index]]
        distance = float(cost[row_index, assignment[row_index]])
        matches.append(
            {
                "different_evidence_image_id": target["image_id"],
                "matched_comparison_image_id": donor["image_id"],
                "standardized_euclidean_distance": distance,
                "different_evidence_features": target["features"],
                "matched_comparison_features": donor["features"],
            }
        )
    return {
        "schema_version": "v4_semantic_covariate_matching_v1",
        "outcome_blind": True,
        "intervention_results_loaded_or_inspected": False,
        "seed": SEED,
        "method": (
            "deterministic minimum-total-cost 1:1 Hungarian assignment without replacement; "
            "row/column order uses SHA256; distance is standardized Euclidean over frozen "
            "question-complexity, answer-length, pair-balance, and image-token features"
        ),
        "features": list(FEATURES),
        "feature_center": {feature: float(center[index]) for index, feature in enumerate(FEATURES)},
        "feature_scale": {feature: float(scale[index]) for index, feature in enumerate(FEATURES)},
        "match_count": len(matches),
        "distance_summary": {
            "median": float(np.median([row["standardized_euclidean_distance"] for row in matches])),
            "q90": float(np.quantile([row["standardized_euclidean_distance"] for row in matches], 0.9)),
            "maximum": max(row["standardized_euclidean_distance"] for row in matches),
        },
        "matches": matches,
    }


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest)
    output = Path(args.output)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    payload = freeze_matching(rows)
    payload["manifest"] = str(manifest)
    payload["manifest_sha256"] = sha256_file(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = output.with_suffix(".sha256")
    checksum.write_text(f"{sha256_file(output)}  {output.name}\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": sha256_file(output), **payload["distance_summary"]}, indent=2))


if __name__ == "__main__":
    main()
