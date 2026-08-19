from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow
import torch
import transformers
import yaml


ACTIONS = ("IGNORE", "READ_ONLY", "WRITE_ONLY", "FULL")
TIE_ORDER = ("FULL", "READ_ONLY", "WRITE_ONLY", "IGNORE")
LAYERS = (0, 4, 8, 12, 16, 20, 24)
CONTRASTS = ("read_w0", "read_w1", "write_r0", "write_r1")
CONTINUOUS = ("four_action_variance", "v_vector_distance", "transfer_regret", "query_oracle_gap")
BINARY = (
    "robust_best_action_disagreement",
    "exact_best_action_disagreement",
    "epsilon_tie_ambiguous",
    "read_w0_sign_reversal",
    "read_w1_sign_reversal",
    "write_r0_sign_reversal",
    "write_r1_sign_reversal",
    "read_w0_silent_to_signed",
    "read_w1_silent_to_signed",
    "write_r0_silent_to_signed",
    "write_r1_silent_to_signed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the frozen v4 GQA discovery sweep.")
    parser.add_argument("--config", default="configs/v4_discovery.yaml")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(base: int, *parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return (base + int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")) % (2**32)


def trimmed_mean(values: np.ndarray, fraction: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    trim = int(math.floor(ordered.size * fraction))
    if trim == 0:
        return float(ordered.mean())
    return float(ordered[trim:-trim].mean())


def summarize(
    values: Iterable[float], bootstrap_replicates: int, bootstrap_seed: int, seed_parts: tuple[Any, ...]
) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size or not np.all(np.isfinite(array)):
        raise ValueError("Summary values must be finite and nonempty")
    generator = np.random.default_rng(stable_seed(bootstrap_seed, *seed_parts))
    draws = generator.integers(0, array.size, size=(bootstrap_replicates, array.size))
    bootstrap_means = array[draws].mean(axis=1)
    return {
        "n_images": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "trimmed_mean_05": trimmed_mean(array, 0.05),
        "trimmed_mean_20": trimmed_mean(array, 0.20),
        "q01": float(np.quantile(array, 0.01)),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "mean_ci_low": float(np.quantile(bootstrap_means, 0.025)),
        "mean_ci_high": float(np.quantile(bootstrap_means, 0.975)),
    }


def sign_label(value: float, epsilon: float) -> str:
    if value > epsilon:
        return "positive"
    if value < -epsilon:
        return "negative"
    return "silent"


def answer_format(answer: str) -> str:
    normalized = str(answer).strip().lower()
    if normalized in {"yes", "no"}:
        return "boolean"
    try:
        float(normalized.replace(",", ""))
        return "numeric"
    except ValueError:
        return "other"


def exact_best(q: dict[str, float]) -> str:
    maximum = max(q.values())
    return next(action for action in TIE_ORDER if q[action] == maximum)


def epsilon_best(q: dict[str, float], epsilon: float) -> tuple[str, ...]:
    maximum = max(q.values())
    return tuple(action for action in ACTIONS if maximum - q[action] <= epsilon)


def effects(q: dict[str, float]) -> dict[str, float]:
    return {
        "read_w0": q["READ_ONLY"] - q["IGNORE"],
        "read_w1": q["FULL"] - q["WRITE_ONLY"],
        "write_r0": q["WRITE_ONLY"] - q["IGNORE"],
        "write_r1": q["FULL"] - q["READ_ONLY"],
        "interaction": q["FULL"] - q["READ_ONLY"] - q["WRITE_ONLY"] + q["IGNORE"],
    }


def merge_and_validate(
    config: dict[str, Any], manifest: list[dict[str, Any]], preflight: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    shard_dir = Path(config["shard_output_dir"])
    for index in range(int(config["shard_count"])):
        completion_path = shard_dir / f"shard_{index:02d}" / "completion.json"
        result_path = shard_dir / f"shard_{index:02d}" / "results.jsonl"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if not completion["complete"] or completion["result_sha256"] != sha256_file(result_path):
            raise RuntimeError(f"Shard {index} is incomplete or has a checksum mismatch")
        rows.extend(read_jsonl(result_path))
    manifest_by_id = {row["id"]: row for row in manifest}
    if len(rows) != 240 or len({row["id"] for row in rows}) != 240:
        raise RuntimeError("Full discovery does not contain exactly 240 unique results")
    if set(manifest_by_id) != {row["id"] for row in rows}:
        raise RuntimeError("Result IDs do not match the frozen manifest")
    if not preflight["gate_pass"]:
        raise RuntimeError("Common-padding preflight gate did not pass")
    maxima = {
        "full_prompt_logit_max_abs": 0.0,
        "full_sequence_score_abs": 0.0,
        "full_mean_score_abs": 0.0,
        "prestate_injection_max_abs": 0.0,
        "read_hook_identity_max_abs": 0.0,
        "write_hook_identity_max_abs": 0.0,
        "answer_token_serialization_sum_abs": 0.0,
    }
    for result in rows:
        source = manifest_by_id[result["id"]]
        for field in ("image_id", "question", "answer", "question_index", "pair_stratum"):
            if result[field] != source[field]:
                raise RuntimeError(f"Frozen metadata mismatch for {result['id']}: {field}")
        if [int(row["layer"]) for row in result["layers"]] != list(LAYERS):
            raise RuntimeError(f"Layer grid mismatch for {result['id']}")
        for layer in result["layers"]:
            if set(layer["states"]) != set(ACTIONS):
                raise RuntimeError(f"Incomplete action matrix for {result['id']} layer {layer['layer']}")
            for action in ACTIONS:
                state = layer["states"][action]
                if not math.isfinite(float(state["sequence_logprob"])) or not math.isfinite(
                    float(state["mean_logprob"])
                ):
                    raise RuntimeError(f"Nonfinite score for {result['id']}")
                for metric in (
                    "prestate_injection_max_abs",
                    "read_hook_identity_max_abs",
                    "write_hook_identity_max_abs",
                ):
                    maxima[metric] = max(maxima[metric], abs(float(state[metric])))
                components = state["accepted_answer_scores"]
                if len(components) != 1:
                    raise RuntimeError(f"GQA must have one accepted answer for {result['id']}")
                component = components[0]
                if (
                    component["answer"] != source["answer"]
                    or component["token_ids"] != source["answer_token_ids"]
                    or int(component["token_length"]) != int(source["answer_token_length"])
                    or float(component["weight"]) != 1.0
                ):
                    raise RuntimeError(f"Accepted-answer target drift for {result['id']}")
                serialized_sum = sum(float(value) for value in component["token_logprobs"])
                serialization_difference = abs(
                    serialized_sum - float(component["sequence_logprob"])
                )
                maxima["answer_token_serialization_sum_abs"] = max(
                    maxima["answer_token_serialization_sum_abs"], serialization_difference
                )
                if serialization_difference > float(
                    config["score_tolerance"]
                ):
                    raise RuntimeError(f"Answer-token sequence aggregation drift for {result['id']}")
            full = layer["states"]["FULL"]
            for source_key, maximum_key in (
                ("baseline_prompt_logit_max_abs", "full_prompt_logit_max_abs"),
                ("baseline_sequence_score_abs", "full_sequence_score_abs"),
                ("baseline_mean_score_abs", "full_mean_score_abs"),
            ):
                maxima[maximum_key] = max(maxima[maximum_key], abs(float(full[source_key])))
    if maxima["prestate_injection_max_abs"] != 0.0:
        raise RuntimeError(f"Pre-layer state identity failed: {maxima}")
    if maxima["full_prompt_logit_max_abs"] > float(config["logit_tolerance"]):
        raise RuntimeError(f"Instrumented FULL logit parity failed: {maxima}")
    if max(maxima["full_sequence_score_abs"], maxima["full_mean_score_abs"]) > float(
        config["score_tolerance"]
    ):
        raise RuntimeError(f"Instrumented FULL score parity failed: {maxima}")
    ordered = sorted(rows, key=lambda row: (int(row["image_index"]), int(row["question_index"])))
    merged = Path(config["merged_output"])
    merged.parent.mkdir(parents=True, exist_ok=True)
    with merged.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return ordered, maxima


def flatten_questions(
    results: list[dict[str, Any]], epsilon: dict[str, float]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        for layer_record in result["layers"]:
            row: dict[str, Any] = {
                "sample_id": result["id"],
                "image_id": str(result["image_id"]),
                "image_index": int(result["image_index"]),
                "question_index": int(result["question_index"]),
                "question": result["question"],
                "answer": result["answer"],
                "answer_format": answer_format(result["answer"]),
                "layer": int(layer_record["layer"]),
                "pair_stratum": result["pair_stratum"],
                "different_evidence": bool(result["different_evidence"]),
                "official_paraphrase": bool(result["official_paraphrase"]),
                "pair_match_distance": float(result["pair_match_distance"]),
                "answer_token_length": int(result["answer_token_length"]),
                "prompt_token_length": int(result["original_prompt_token_length"]),
                "common_prompt_token_length": int(result["common_prompt_token_length"]),
                "visual_token_count": int(result["visual_token_count"]),
                "semantic_program_depth": int(result["semantic_program_depth"]),
                "question_structural_type": (result.get("question_types") or {}).get("structural"),
                "question_semantic_type": (result.get("question_types") or {}).get("semantic"),
                "question_detailed_type": (result.get("question_types") or {}).get("detailed"),
            }
            for metric in ("mean", "sequence"):
                q = {
                    action: float(layer_record["states"][action][f"{metric}_logprob"])
                    for action in ACTIONS
                }
                for action in ACTIONS:
                    row[f"q_{metric}_{action.lower()}"] = q[action]
                    row[f"v_{metric}_{action.lower()}"] = q[action] - q["FULL"]
                derived = effects(q)
                for name, value in derived.items():
                    row[f"{name}_{metric}"] = value
                best = epsilon_best(q, epsilon[metric])
                row[f"epsilon_best_{metric}"] = "|".join(best)
                row[f"epsilon_best_size_{metric}"] = len(best)
                row[f"exact_best_{metric}"] = exact_best(q)
                row[f"q_range_{metric}"] = max(q.values()) - min(q.values())
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["image_index", "question_index", "layer"])
    if len(frame) != 1680:
        raise RuntimeError("Expected 1,680 complete question-layer matrices")
    return frame.reset_index(drop=True)


def build_image_layer(question_frame: pd.DataFrame, epsilon: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (image_id, layer), group in question_frame.groupby(["image_id", "layer"], sort=True):
        group = group.sort_values("question_index")
        if len(group) != 2:
            raise RuntimeError(f"Image {image_id} layer {layer} does not have two questions")
        output: dict[str, Any] = {
            "image_id": image_id,
            "image_index": int(group.iloc[0]["image_index"]),
            "layer": int(layer),
            "pair_stratum": group.iloc[0]["pair_stratum"],
            "different_evidence": bool(group.iloc[0]["different_evidence"]),
            "official_paraphrase": bool(group.iloc[0]["official_paraphrase"]),
            "pair_match_distance": float(group.iloc[0]["pair_match_distance"]),
            "answer_token_length_difference": abs(
                int(group.iloc[0]["answer_token_length"])
                - int(group.iloc[1]["answer_token_length"])
            ),
            "prompt_token_length_difference": abs(
                int(group.iloc[0]["prompt_token_length"])
                - int(group.iloc[1]["prompt_token_length"])
            ),
            "program_depth_difference": abs(
                int(group.iloc[0]["semantic_program_depth"])
                - int(group.iloc[1]["semantic_program_depth"])
            ),
            "question_type_mismatch_count": sum(
                group.iloc[0][field] != group.iloc[1][field]
                for field in (
                    "question_structural_type",
                    "question_semantic_type",
                    "question_detailed_type",
                )
            ),
            "answer_format_equal": bool(
                group.iloc[0]["answer_format"] == group.iloc[1]["answer_format"]
            ),
        }
        for metric in ("mean", "sequence"):
            q_values = []
            v_values = []
            best_sets = []
            exact = []
            contrast_labels: list[dict[str, str]] = []
            for _, question in group.iterrows():
                q = {action: float(question[f"q_{metric}_{action.lower()}"]) for action in ACTIONS}
                q_values.append(q)
                v_values.append(np.asarray([q[action] - q["FULL"] for action in ACTIONS]))
                best_sets.append(set(str(question[f"epsilon_best_{metric}"]).split("|")))
                exact.append(str(question[f"exact_best_{metric}"]))
                contrast_labels.append(
                    {
                        contrast: sign_label(float(question[f"{contrast}_{metric}"]), epsilon[metric])
                        for contrast in CONTRASTS
                    }
                )
            output[f"robust_best_action_disagreement_{metric}"] = int(
                best_sets[0].isdisjoint(best_sets[1])
            )
            output[f"exact_best_action_disagreement_{metric}"] = int(exact[0] != exact[1])
            output[f"epsilon_tie_ambiguous_{metric}"] = int(
                len(best_sets[0]) > 1 or len(best_sets[1]) > 1
            )
            output[f"question0_exact_best_{metric}"] = exact[0]
            output[f"question1_exact_best_{metric}"] = exact[1]
            output[f"question0_epsilon_best_{metric}"] = "|".join(sorted(best_sets[0]))
            output[f"question1_epsilon_best_{metric}"] = "|".join(sorted(best_sets[1]))
            for contrast in CONTRASTS:
                labels = {contrast_labels[0][contrast], contrast_labels[1][contrast]}
                output[f"{contrast}_sign_reversal_{metric}"] = int(
                    labels == {"positive", "negative"}
                )
                output[f"{contrast}_silent_to_signed_{metric}"] = int(
                    "silent" in labels and len(labels) == 2
                )
            values = np.asarray(v_values, dtype=np.float64)
            output[f"four_action_variance_{metric}"] = float(
                np.var(values, axis=0, ddof=0).mean()
            )
            output[f"v_vector_distance_{metric}"] = float(np.linalg.norm(values[0] - values[1]))
            transfer = []
            for source, target, source_best in (
                (q_values[0], q_values[1], best_sets[0]),
                (q_values[1], q_values[0], best_sets[1]),
            ):
                del source
                transfer.append(max(target.values()) - max(target[action] for action in source_best))
            output[f"transfer_regret_{metric}"] = float(np.mean(transfer))
            query_oracle = np.mean([max(row) for row in values])
            image_oracle = max(values.mean(axis=0))
            output[f"query_oracle_gap_{metric}"] = float(query_oracle - image_oracle)
        rows.append(output)
    frame = pd.DataFrame(rows).sort_values(["image_index", "layer"]).reset_index(drop=True)
    if len(frame) != 840:
        raise RuntimeError("Expected 840 image-layer records")
    return frame


def expanded_scopes(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [
        ("all", frame),
        ("different_evidence", frame[frame.pair_stratum == "different_evidence"]),
        ("matched_comparison", frame[frame.pair_stratum == "matched_comparison"]),
    ]


def build_summaries(
    image_layer: pd.DataFrame, replicates: int, seed: int
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    equal_layer = (
        image_layer.groupby(["image_id", "image_index", "pair_stratum"], as_index=False)
        .agg(
            {
                column: "mean"
                for column in image_layer.columns
                if any(column.startswith(f"{name}_") for name in CONTINUOUS + BINARY)
            }
        )
        .sort_values("image_index")
    )
    summary_rows: list[dict[str, Any]] = []
    for layer_label, frame in [(str(layer), image_layer[image_layer.layer == layer]) for layer in LAYERS] + [
        ("equal_layer_average", equal_layer)
    ]:
        for scope, part in expanded_scopes(frame):
            for metric in ("mean", "sequence"):
                for quantity in CONTINUOUS + BINARY:
                    column = f"{quantity}_{metric}"
                    summary_rows.append(
                        {
                            "layer": layer_label,
                            "scope": scope,
                            "utility": metric,
                            "quantity": quantity,
                            **summarize(part[column], replicates, seed, (layer_label, scope, metric, quantity)),
                        }
                    )
    return summary_rows, equal_layer


def fit_adjusted_effect(
    outcome: np.ndarray, treatment: np.ndarray, covariates: np.ndarray
) -> float:
    design = np.column_stack([np.ones(outcome.size), treatment, covariates])
    coefficients, _, _, _ = np.linalg.lstsq(design, outcome, rcond=None)
    return float(coefficients[1])


def semantic_sensitivity(
    equal_layer: pd.DataFrame,
    matching: dict[str, Any],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    metadata: dict[str, dict[str, float]] = {}
    pairs = []
    for match in matching["matches"]:
        left = str(match["different_evidence_image_id"])
        right = str(match["matched_comparison_image_id"])
        metadata[left] = match["different_evidence_features"]
        metadata[right] = match["matched_comparison_features"]
        pairs.append((left, right))
    indexed = equal_layer.set_index("image_id")
    features = list(matching["features"])
    center = np.asarray([matching["feature_center"][name] for name in features])
    scale = np.asarray([matching["feature_scale"][name] for name in features])
    ordered_ids = sorted(indexed.index)
    covariates = np.asarray(
        [[metadata[image_id][name] for name in features] for image_id in ordered_ids], dtype=np.float64
    )
    covariates = (covariates - center) / scale
    treatment = np.asarray(
        [indexed.loc[image_id, "pair_stratum"] == "different_evidence" for image_id in ordered_ids],
        dtype=np.float64,
    )
    results = []
    for metric in ("mean", "sequence"):
        for quantity in (
            "robust_best_action_disagreement",
            "v_vector_distance",
            "transfer_regret",
            "query_oracle_gap",
        ):
            column = f"{quantity}_{metric}"
            paired = np.asarray(
                [float(indexed.loc[left, column] - indexed.loc[right, column]) for left, right in pairs]
            )
            adjusted_outcome = np.asarray(
                [float(indexed.loc[image_id, column]) for image_id in ordered_ids], dtype=np.float64
            )
            coefficient = fit_adjusted_effect(adjusted_outcome, treatment, covariates)
            generator = np.random.default_rng(stable_seed(seed, "semantic", metric, quantity))
            draw_indices = generator.integers(0, len(pairs), size=(replicates, len(pairs)))
            paired_boot = paired[draw_indices].mean(axis=1)
            results.append(
                {
                    "utility": metric,
                    "quantity": quantity,
                    "paired_different_minus_comparison_mean": float(paired.mean()),
                    "paired_difference_ci_low": float(np.quantile(paired_boot, 0.025)),
                    "paired_difference_ci_high": float(np.quantile(paired_boot, 0.975)),
                    "covariate_adjusted_stratum_coefficient": coefficient,
                }
            )
    return {
        "paraphrase_control_available": False,
        "paraphrase_image_count": 0,
        "semantic_claim_gate_evaluable": False,
        "limitation": (
            "The frozen 240-question natural GQA pool contains no official equivalent pairs "
            "meeting the prospective answer/type/target rule; no generated paraphrases were added."
        ),
        "matching_method": matching["method"],
        "matching_distance_summary": matching["distance_summary"],
        "adjustment_features": features,
        "results": results,
    }


def heavy_tail_sensitivity(equal_layer: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for metric in ("mean", "sequence"):
        for quantity in CONTINUOUS:
            values = np.asarray(equal_layer[f"{quantity}_{metric}"], dtype=np.float64)
            ordered = np.sort(values)[::-1]
            total = float(ordered.sum())
            top_count = max(1, int(math.ceil(0.05 * ordered.size)))
            rows.append(
                {
                    "utility": metric,
                    "quantity": quantity,
                    "maximum": float(ordered[0]),
                    "top_one_share_of_sum": float(ordered[0] / total) if total > 0 else 0.0,
                    "top_five_percent_share_of_sum": float(ordered[:top_count].sum() / total)
                    if total > 0
                    else 0.0,
                    "positive_image_fraction": float(np.mean(values > 0)),
                    "at_least_0_05_fraction": float(np.mean(values >= 0.05))
                    if metric == "mean"
                    else math.nan,
                }
            )
    return rows


def robustness_sensitivity(
    image_layer: pd.DataFrame,
    equal_layer: pd.DataFrame,
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    metadata = (
        image_layer.sort_values("layer")
        .groupby("image_id", as_index=False)
        .first()[
            [
                "image_id",
                "answer_token_length_difference",
                "program_depth_difference",
                "question_type_mismatch_count",
                "answer_format_equal",
            ]
        ]
    )
    any_tie = (
        image_layer.groupby("image_id")["epsilon_tie_ambiguous_mean"].max().rename("any_tie")
    )
    metadata = metadata.merge(any_tie, on="image_id", how="left")
    frame = equal_layer.merge(metadata, on="image_id", how="left")
    masks = {
        "complete": np.ones(len(frame), dtype=bool),
        "all_layers_tie_free": frame["any_tie"].to_numpy() == 0,
        "answer_length_equal": frame["answer_token_length_difference"].to_numpy() == 0,
        "answer_format_equal": frame["answer_format_equal"].to_numpy(dtype=bool),
        "strict_difficulty_proxy_match": (
            (frame["program_depth_difference"].to_numpy() == 0)
            & (frame["question_type_mismatch_count"].to_numpy() == 0)
        ),
    }
    masks["joint_strict_match"] = (
        masks["answer_length_equal"]
        & masks["answer_format_equal"]
        & masks["strict_difficulty_proxy_match"]
    )
    rows = []
    for subset, mask in masks.items():
        part = frame.loc[mask]
        if part.empty:
            rows.append({"subset": subset, "n_images": 0})
            continue
        for quantity in CONTINUOUS + ("robust_best_action_disagreement",):
            values = np.asarray(part[f"{quantity}_mean"], dtype=np.float64)
            summary = summarize(values, replicates, seed, ("robustness", subset, quantity))
            lower = float(np.quantile(values, 0.05))
            upper = float(np.quantile(values, 0.95))
            summary["winsorized_05_mean"] = float(np.clip(values, lower, upper).mean())
            top_count = max(1, int(math.ceil(values.size * 0.05)))
            summary["mean_excluding_top_05"] = float(np.sort(values)[:-top_count].mean()) if values.size > top_count else math.nan
            rows.append({"subset": subset, "quantity": quantity, **summary})
    return rows


def action_frequency(question_frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for layer in LAYERS:
        part = question_frame[question_frame.layer == layer]
        for metric in ("mean", "sequence"):
            counts = Counter(part[f"exact_best_{metric}"])
            for action in ACTIONS:
                rows.append(
                    {
                        "layer": layer,
                        "utility": metric,
                        "action": action,
                        "count": counts[action],
                        "fraction": counts[action] / len(part),
                    }
                )
    return rows


def metric_agreement(image_layer: pd.DataFrame, equal_layer: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for layer_label, frame in [(str(layer), image_layer[image_layer.layer == layer]) for layer in LAYERS] + [
        ("equal_layer_average", equal_layer)
    ]:
        for quantity in CONTINUOUS:
            left = np.asarray(frame[f"{quantity}_mean"], dtype=np.float64)
            right = np.asarray(frame[f"{quantity}_sequence"], dtype=np.float64)
            pearson = float(np.corrcoef(left, right)[0, 1]) if left.std() > 0 and right.std() > 0 else math.nan
            rows.append(
                {
                    "layer": layer_label,
                    "quantity": quantity,
                    "pearson_per_token_vs_sequence": pearson,
                    "positive_label_agreement": float(np.mean((left > 0) == (right > 0))),
                }
            )
        for quantity in BINARY:
            left = np.asarray(frame[f"{quantity}_mean"], dtype=np.float64)
            right = np.asarray(frame[f"{quantity}_sequence"], dtype=np.float64)
            rows.append(
                {
                    "layer": layer_label,
                    "quantity": quantity,
                    "pearson_per_token_vs_sequence": math.nan,
                    "positive_label_agreement": float(np.mean(left == right)),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = Path(config["manifest"])
    checksum_path = Path(config["manifest_checksum"])
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != checksum_path.read_text(encoding="utf-8").split()[0]:
        raise RuntimeError("Frozen manifest checksum mismatch")
    manifest = read_jsonl(manifest_path)
    preflight_path = Path(config["preflight_output_dir"]) / "v4_common_padding_preflight_v1.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    epsilon = {
        "mean": float(preflight["noise"]["epsilon_mean"]),
        "sequence": float(preflight["noise"]["epsilon_sequence"]),
    }
    results, integrity = merge_and_validate(config, manifest, preflight)
    question_frame = flatten_questions(results, epsilon)
    image_layer = build_image_layer(question_frame, epsilon)
    summary_rows, equal_layer = build_summaries(
        image_layer, int(config["bootstrap_replicates"]), int(config["bootstrap_seed"])
    )
    matching_path = Path(
        "outputs/v4_discovery/manifest/v4_semantic_covariate_matching_v1.json"
    )
    matching = json.loads(matching_path.read_text(encoding="utf-8"))
    if matching["manifest_sha256"] != manifest_sha:
        raise RuntimeError("Semantic matching was not frozen against this discovery manifest")
    semantic = semantic_sensitivity(
        equal_layer, matching, int(config["bootstrap_replicates"]), int(config["bootstrap_seed"])
    )

    output_dir = Path(config["analysis_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    question_path = output_dir / "question_layer_q_v_v1.parquet"
    image_layer_path = output_dir / "image_layer_query_dependence_v1.csv"
    equal_layer_path = output_dir / "image_equal_layer_summary_v1.csv"
    summary_path = output_dir / "image_clustered_summaries_v1.csv"
    action_path = output_dir / "best_action_frequencies_v1.csv"
    semantic_path = output_dir / "semantic_control_sensitivity_v1.json"
    tail_path = output_dir / "heavy_tail_sensitivity_v1.csv"
    agreement_path = output_dir / "sequence_per_token_agreement_v1.csv"
    robustness_path = output_dir / "robustness_sensitivity_v1.csv"
    question_frame.to_parquet(question_path, index=False)
    image_layer.to_csv(image_layer_path, index=False)
    equal_layer.to_csv(equal_layer_path, index=False)
    write_csv(summary_path, summary_rows)
    write_csv(action_path, action_frequency(question_frame))
    write_json(semantic_path, semantic)
    write_csv(tail_path, heavy_tail_sensitivity(equal_layer))
    write_csv(agreement_path, metric_agreement(image_layer, equal_layer))
    write_csv(
        robustness_path,
        robustness_sensitivity(
            image_layer,
            equal_layer,
            int(config["bootstrap_replicates"]),
            int(config["bootstrap_seed"]),
        ),
    )

    artifacts = [
        Path(config["merged_output"]),
        question_path,
        image_layer_path,
        equal_layer_path,
        summary_path,
        action_path,
        semantic_path,
        tail_path,
        agreement_path,
        robustness_path,
    ]
    analysis_manifest = {
        "schema_version": "v4_gqa_discovery_analysis_manifest_v1",
        "manifest_sha256": manifest_sha,
        "v4_plan_sha256": sha256_file(Path("workspace/dynamic_mllm_query_conditional_plan_v4.md")),
        "preflight_sha256": sha256_file(preflight_path),
        "semantic_matching_sha256": sha256_file(matching_path),
        "config_sha256": sha256_file(config_path),
        "model_config_sha256": sha256_file(Path(config["model_config"])),
        "layers": list(LAYERS),
        "actions": list(ACTIONS),
        "image_count": 120,
        "question_count": 240,
        "question_layer_count": len(question_frame),
        "image_layer_count": len(image_layer),
        "epsilon": epsilon,
        "practical_threshold_mean": float(config["practical_threshold_mean"]),
        "bootstrap": {
            "unit": "image",
            "replicates": int(config["bootstrap_replicates"]),
            "seed": int(config["bootstrap_seed"]),
            "interval": "percentile 95%",
        },
        "integrity": integrity,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "artifacts": {str(path): sha256_file(path) for path in artifacts},
        "complete": True,
    }
    write_json(output_dir / "analysis_manifest.json", analysis_manifest)
    print(
        json.dumps(
            {
                "complete": True,
                "records": len(results),
                "question_layer": len(question_frame),
                "image_layer": len(image_layer),
                "epsilon": epsilon,
                "analysis_manifest": str(output_dir / "analysis_manifest.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
