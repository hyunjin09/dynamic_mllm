from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge and analyze frozen query-refinement discovery.")
    parser.add_argument("--config", default="configs/query_refinement_gqa.yaml")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trimmed_mean(values: np.ndarray, fraction: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    cut = int(math.floor(fraction * len(ordered)))
    if 2 * cut >= len(ordered):
        return float("nan")
    return float(ordered[cut : len(ordered) - cut].mean())


def cluster_bootstrap(values: np.ndarray, images: np.ndarray, draws: int, seed: int):
    unique = np.unique(images)
    by_image = {image: values[images == image] for image in unique}
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    fractions = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        concatenated = np.concatenate([by_image[image] for image in sampled])
        means[draw] = concatenated.mean()
        fractions[draw] = (concatenated > 0).mean()
    return {
        "mean_ci95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "positive_fraction_ci95": [
            float(value) for value in np.quantile(fractions, [0.025, 0.975])
        ],
    }


def summarize(values: np.ndarray, images: np.ndarray, config: dict[str, Any], seed_offset: int):
    values = np.asarray(values, dtype=np.float64)
    bootstrap = cluster_bootstrap(
        values,
        images,
        int(config["bootstrap_replicates"]),
        int(config["bootstrap_seed"]) + seed_offset,
    )
    count_remove = max(1, int(math.ceil(0.05 * len(values))))
    retained = np.delete(values, np.argsort(np.abs(values))[-count_remove:])
    result = {
        "count": len(values),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "trimmed_mean_20": trimmed_mean(values, 0.2),
        "quantiles": {
            str(q): float(np.quantile(values, q))
            for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
        },
        "positive_fraction": float((values > 0).mean()),
        "practical_positive_fraction": float(
            (values >= float(config["practical_effect_threshold_mean"])).mean()
        ),
        "top_abs_5pct_removed_mean": float(retained.mean()),
        **bootstrap,
    }
    result["robust_pass"] = bool(
        result["mean"] >= float(config["practical_effect_threshold_mean"])
        and result["mean_ci95"][0] > 0
        and result["median"] > 0
        and result["trimmed_mean_20"] > 0
        and result["positive_fraction"] > float(config["positive_fraction_min"])
        and result["top_abs_5pct_removed_mean"] > 0
    )
    return result


def behavior_counts(frame: pd.DataFrame, comparator: str) -> dict[str, int]:
    target = frame["target_correct"].to_numpy() > 0
    control = frame[f"{comparator}_correct"].to_numpy() > 0
    return {
        "control_wrong_to_target_correct": int((~control & target).sum()),
        "control_correct_to_target_wrong": int((control & ~target).sum()),
        "unchanged_correct": int((control & target).sum()),
        "unchanged_wrong_or_different_wrong": int((~control & ~target).sum()),
        "no_net_regression": bool((~control & target).sum() >= (control & ~target).sum()),
    }


def safe_spearman(first: pd.Series, second: pd.Series) -> float | None:
    paired = pd.DataFrame({"first": first, "second": second}).dropna()
    if paired["first"].nunique() < 2 or paired["second"].nunique() < 2:
        return None
    # Spearman correlation is Pearson correlation of average ranks. Computing
    # it directly avoids adding SciPy to the frozen project environment.
    first_rank = paired["first"].rank(method="average").to_numpy(dtype=np.float64)
    second_rank = paired["second"].rank(method="average").to_numpy(dtype=np.float64)
    value = np.corrcoef(first_rank, second_rank)[0, 1]
    return None if np.isnan(value) else float(value)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rows = []
    completion_paths = []
    result_paths = []
    for shard in range(int(config["shard_count"])):
        shard_dir = Path(config["shard_output_dir"]) / f"shard_{shard:02d}"
        completion_path = shard_dir / "completion.json"
        result_path = shard_dir / "results.jsonl"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if not completion["complete"] or sha256_file(result_path) != completion["result_sha256"]:
            raise RuntimeError(f"Incomplete or checksum-invalid shard {shard}")
        rows.extend(read_jsonl(result_path))
        completion_paths.append(completion_path)
        result_paths.append(result_path)
    rows.sort(key=lambda row: (int(row["image_index"]), int(row["question_index"])))
    if len(rows) != int(config["sample_count"]):
        raise RuntimeError(f"Expected {config['sample_count']} rows, found {len(rows)}")
    if len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate sample IDs in merged results")
    if len({row["image_id"] for row in rows}) != int(config["image_count"]):
        raise RuntimeError("Merged unique-image count drift")
    merged_path = Path(config["merged_jsonl"])
    write_jsonl(merged_path, rows)

    q_rows = []
    contrast_rows = []
    for row in rows:
        baseline = row["BASELINE"]
        for layer_data in row["layers"]:
            layer = int(layer_data["layer"])
            variants = layer_data["variants"]
            unconditioned = variants["UNCONDITIONED_REPLAY"]
            target = variants["TARGET_QUERY_REPLAY"]
            other = variants["OTHER_QUERY_REPLAY"]
            q_row = {
                "id": row["id"],
                "image_id": row["image_id"],
                "question_index": row["question_index"],
                "paired_other_id": row["paired_other_id"],
                "layer": layer,
                "question": row["question"],
                "answer": row["answer"],
                "pair_stratum": row["pair_stratum"],
                "different_evidence": row["different_evidence"],
                "question_types_json": json.dumps(row["question_types"], sort_keys=True),
                "semantic_object_ids_json": json.dumps(row["semantic_object_ids"]),
                "question_word_length": row["question_word_length"],
                "answer_token_length": row["answer_token_length"],
                "prompt_token_length": row["prompt_token_length"],
                "visual_token_count": row["visual_token_count"],
                "baseline_mean": baseline["mean_logprob"],
                "unconditioned_mean": unconditioned["mean_logprob"],
                "target_mean": target["mean_logprob"],
                "other_mean": other["mean_logprob"],
                "baseline_sequence": baseline["sequence_logprob"],
                "unconditioned_sequence": unconditioned["sequence_logprob"],
                "target_sequence": target["sequence_logprob"],
                "other_sequence": other["sequence_logprob"],
                "baseline_generated": baseline["generated_answer"],
                "unconditioned_generated": unconditioned["generated_answer"],
                "target_generated": target["generated_answer"],
                "other_generated": other["generated_answer"],
                "baseline_correct": baseline["official_correctness"],
                "unconditioned_correct": unconditioned["official_correctness"],
                "target_correct": target["official_correctness"],
                "other_correct": other["official_correctness"],
            }
            q_rows.append(q_row)
            contrast_rows.append(
                {
                    **q_row,
                    "delta_condition_mean": target["mean_logprob"]
                    - unconditioned["mean_logprob"],
                    "delta_target_mean": target["mean_logprob"] - other["mean_logprob"],
                    "delta_base_mean": target["mean_logprob"] - baseline["mean_logprob"],
                    "delta_unconditioned_base_mean": unconditioned["mean_logprob"]
                    - baseline["mean_logprob"],
                    "delta_condition_sequence": target["sequence_logprob"]
                    - unconditioned["sequence_logprob"],
                    "delta_target_sequence": target["sequence_logprob"]
                    - other["sequence_logprob"],
                    "delta_base_sequence": target["sequence_logprob"]
                    - baseline["sequence_logprob"],
                }
            )
    q_frame = pd.DataFrame(q_rows)
    contrast_frame = pd.DataFrame(contrast_rows)
    q_path = Path(config["q_parquet"])
    q_path.parent.mkdir(parents=True, exist_ok=True)
    q_frame.to_parquet(q_path, index=False)
    contrast_path = Path(config["contrasts_csv"])
    contrast_frame.to_csv(contrast_path, index=False)

    layer_summaries = []
    behavior = {}
    control_correlations = []
    for layer_index, layer in enumerate(config["refinement_layers"]):
        frame = contrast_frame[contrast_frame["layer"] == int(layer)].copy()
        images = frame["image_id"].astype(str).to_numpy()
        condition = summarize(
            frame["delta_condition_mean"].to_numpy(), images, config, 10 * layer_index
        )
        target = summarize(
            frame["delta_target_mean"].to_numpy(), images, config, 10 * layer_index + 1
        )
        extra = summarize(
            frame["delta_unconditioned_base_mean"].to_numpy(),
            images,
            config,
            10 * layer_index + 2,
        )
        baseline = summarize(
            frame["delta_base_mean"].to_numpy(), images, config, 10 * layer_index + 3
        )
        layer_behavior = {
            comparator: behavior_counts(frame, comparator)
            for comparator in ("baseline", "unconditioned", "other")
        }
        behavior[str(layer)] = layer_behavior
        behavior_pass = all(value["no_net_regression"] for value in layer_behavior.values())
        layer_pass = condition["robust_pass"] and target["robust_pass"] and behavior_pass
        layer_summaries.append(
            {
                "layer": int(layer),
                "delta_condition": condition,
                "delta_target": target,
                "unconditioned_minus_baseline": extra,
                "target_minus_baseline": baseline,
                "behavior_pass": behavior_pass,
                "discovery_layer_pass": layer_pass,
            }
        )
        for outcome in ("delta_condition_mean", "delta_target_mean"):
            for covariate in (
                "question_word_length",
                "answer_token_length",
                "baseline_mean",
                "prompt_token_length",
                "visual_token_count",
            ):
                control_correlations.append(
                    {
                        "layer": int(layer),
                        "outcome": outcome,
                        "covariate": covariate,
                        "spearman": safe_spearman(frame[outcome], frame[covariate]),
                    }
                )

    within_rows = []
    for (image_id, layer), group in contrast_frame.groupby(["image_id", "layer"], sort=True):
        if len(group) != 2:
            raise RuntimeError("Every image-layer group must have two questions")
        ordered = group.sort_values("question_index")
        first, second = ordered.iloc[0], ordered.iloc[1]
        first_objects = set(json.loads(first["semantic_object_ids_json"]))
        second_objects = set(json.loads(second["semantic_object_ids_json"]))
        union = first_objects | second_objects
        object_jaccard_distance = (
            1.0 - len(first_objects & second_objects) / len(union) if union else None
        )
        within_rows.append(
            {
                "image_id": image_id,
                "layer": int(layer),
                "absolute_within_image_delta_condition_difference": abs(
                    first["delta_condition_mean"] - second["delta_condition_mean"]
                ),
                "signed_within_image_delta_condition_difference": first[
                    "delta_condition_mean"
                ]
                - second["delta_condition_mean"],
                "object_jaccard_distance": object_jaccard_distance,
                "different_evidence": bool(first["different_evidence"]),
                "question_length_difference": abs(
                    first["question_word_length"] - second["question_word_length"]
                ),
                "answer_length_difference": abs(
                    first["answer_token_length"] - second["answer_token_length"]
                ),
                "question_type_same": first["question_types_json"]
                == second["question_types_json"],
            }
        )
    within_frame = pd.DataFrame(within_rows)
    pair_summary = []
    for layer in config["refinement_layers"]:
        frame = within_frame[within_frame["layer"] == int(layer)]
        pair_summary.append(
            {
                "layer": int(layer),
                "mean_absolute_within_image_delta_condition_difference": float(
                    frame["absolute_within_image_delta_condition_difference"].mean()
                ),
                "semantic_object_distance_spearman": safe_spearman(
                    frame["absolute_within_image_delta_condition_difference"],
                    frame["object_jaccard_distance"],
                ),
                "question_length_difference_spearman": safe_spearman(
                    frame["absolute_within_image_delta_condition_difference"],
                    frame["question_length_difference"],
                ),
                "different_evidence_mean": float(
                    frame.loc[
                        frame["different_evidence"],
                        "absolute_within_image_delta_condition_difference",
                    ].mean()
                ),
                "matched_comparison_mean": float(
                    frame.loc[
                        ~frame["different_evidence"],
                        "absolute_within_image_delta_condition_difference",
                    ].mean()
                ),
            }
        )

    question_type_summary = []
    for layer in config["refinement_layers"]:
        layer_frame = contrast_frame[contrast_frame["layer"] == int(layer)].copy()
        parsed_types = layer_frame["question_types_json"].map(json.loads)
        for dimension in ("structural", "semantic", "detailed"):
            layer_frame[dimension] = parsed_types.map(
                lambda value: str((value or {}).get(dimension) or "missing")
            )
            for category, group in layer_frame.groupby(dimension, sort=True):
                question_type_summary.append(
                    {
                        "layer": int(layer),
                        "dimension": dimension,
                        "category": category,
                        "count": len(group),
                        "delta_condition_mean": float(group["delta_condition_mean"].mean()),
                        "delta_condition_median": float(
                            group["delta_condition_mean"].median()
                        ),
                        "delta_target_mean": float(group["delta_target_mean"].mean()),
                        "delta_target_median": float(group["delta_target_mean"].median()),
                    }
                )

    successful_layers = [
        row["layer"] for row in layer_summaries if row["discovery_layer_pass"]
    ]
    decision = (
        "PROCEED_TO_QUERY_REFINEMENT_CONFIRMATION"
        if len(successful_layers) >= int(config["minimum_successful_layers"])
        else "STOP_QUERY_REFINEMENT_DIRECTION"
    )
    output_dir = Path(config["analysis_manifest"]).parent / "analysis_v1"
    write_json(output_dir / "layer_summaries.json", layer_summaries)
    write_json(output_dir / "behavior_counts.json", behavior)
    write_json(output_dir / "question_pair_dependence.json", pair_summary)
    write_json(output_dir / "question_type_summary.json", question_type_summary)
    write_json(output_dir / "descriptive_covariate_correlations.json", control_correlations)
    analysis_manifest = {
        "schema_version": "query_refinement_analysis_manifest_v1",
        "integrity_status": "pass",
        "completion": {
            "records": len(rows),
            "unique_images": len({row["image_id"] for row in rows}),
            "sample_layer_rows": len(contrast_frame),
            "no_sample_replacement": True,
        },
        "frozen_settings": {
            "layers": config["refinement_layers"],
            "variants": config["variants"],
            "bootstrap_replicates": config["bootstrap_replicates"],
            "bootstrap_seed": config["bootstrap_seed"],
            "practical_effect_threshold_mean": config[
                "practical_effect_threshold_mean"
            ],
            "positive_fraction_min": config["positive_fraction_min"],
            "minimum_successful_layers": config["minimum_successful_layers"],
        },
        "successful_layers": successful_layers,
        "decision": decision,
        "checksums": {
            str(config_path): sha256_file(config_path),
            str(config["manifest"]): sha256_file(Path(config["manifest"])),
            str(config["manifest_audit"]): sha256_file(Path(config["manifest_audit"])),
            str(Path(config["preflight_output_dir"]) / "summary.json"): sha256_file(
                Path(config["preflight_output_dir"]) / "summary.json"
            ),
            str(merged_path): sha256_file(merged_path),
            str(q_path): sha256_file(q_path),
            str(contrast_path): sha256_file(contrast_path),
            **{str(path): sha256_file(path) for path in result_paths + completion_paths},
        },
        "artifact_paths": {
            "q_parquet": str(q_path),
            "contrasts_csv": str(contrast_path),
            "layer_summaries": str(output_dir / "layer_summaries.json"),
            "behavior_counts": str(output_dir / "behavior_counts.json"),
            "question_pair_dependence": str(output_dir / "question_pair_dependence.json"),
            "question_type_summary": str(output_dir / "question_type_summary.json"),
            "covariate_correlations": str(
                output_dir / "descriptive_covariate_correlations.json"
            ),
        },
    }
    write_json(Path(config["analysis_manifest"]), analysis_manifest)
    print(json.dumps(analysis_manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
