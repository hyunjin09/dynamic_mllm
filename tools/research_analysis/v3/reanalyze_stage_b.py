from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc


ACTIONS = ("IGNORE", "READ_ONLY", "WRITE_ONLY", "FULL")
SUPPRESSIONS = ("IGNORE", "READ_ONLY", "WRITE_ONLY")
ACTION_BITS = {
    "IGNORE": (0, 0),
    "READ_ONLY": (1, 0),
    "WRITE_ONLY": (0, 1),
    "FULL": (1, 1),
}
BITS_ACTION = {bits: action for action, bits in ACTION_BITS.items()}
TIE_PREFERENCE = ("FULL", "READ_ONLY", "WRITE_ONLY", "IGNORE")
METRICS = ("mean", "sequence")
LAYERS = (0, 4, 8, 12, 16, 20, 24, 27)
EPSILON = {"mean": 1e-6, "sequence": 1e-5}
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 2026080601
CORRECTNESS_THRESHOLD = 1.0
PRACTICAL_NEAR_TIE_MEAN = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reanalyze frozen Stage B under plan v3.")
    parser.add_argument("--results", default="outputs/stage_b/stage_b_results_v1.jsonl")
    parser.add_argument("--candidate-manifest", default="data_manifests/stage_b_discovery_candidates_400.jsonl")
    parser.add_argument("--output-dir", default="outputs/v3_discovery")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return (BOOTSTRAP_SEED + int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")) % (2**32)


def trimmed_mean(values: np.ndarray, fraction: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    trim = int(math.floor(ordered.size * fraction))
    if trim == 0:
        return float(ordered.mean())
    if trim * 2 >= ordered.size:
        raise ValueError("Trim fraction removes every observation")
    return float(ordered[trim:-trim].mean())


def summarize(values: Iterable[float], seed_parts: tuple[Any, ...]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size or not np.all(np.isfinite(array)):
        raise ValueError("Summary requires finite nonempty values")
    generator = np.random.default_rng(stable_seed(*seed_parts))
    draws = generator.integers(0, array.size, size=(BOOTSTRAP_REPLICATES, array.size))
    boot = array[draws].mean(axis=1)
    return {
        "n": int(array.size),
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
        "mean_ci_low": float(np.quantile(boot, 0.025)),
        "mean_ci_high": float(np.quantile(boot, 0.975)),
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def correlation(left: Iterable[float], right: Iterable[float], rank: bool = False) -> float:
    x = np.asarray(list(left), dtype=np.float64)
    y = np.asarray(list(right), dtype=np.float64)
    if rank:
        x, y = average_ranks(x), average_ranks(y)
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def sign_label(value: float, epsilon: float) -> str:
    if value > epsilon:
        return "positive"
    if value < -epsilon:
        return "negative"
    return "silent"


def conservative_best(q: Mapping[str, float], epsilon: float) -> tuple[str, tuple[str, ...]]:
    maximum = max(float(value) for value in q.values())
    ties = tuple(action for action in ACTIONS if float(q[action]) >= maximum - epsilon)
    selected = next(action for action in TIE_PREFERENCE if action in ties)
    return selected, ties


def exact_best(q: Mapping[str, float]) -> str:
    maximum = max(float(value) for value in q.values())
    return next(action for action in TIE_PREFERENCE if float(q[action]) == maximum)


def independent_action(q: Mapping[str, float], epsilon: float) -> str:
    read_main = 0.5 * ((q["READ_ONLY"] - q["IGNORE"]) + (q["FULL"] - q["WRITE_ONLY"]))
    write_main = 0.5 * ((q["WRITE_ONLY"] - q["IGNORE"]) + (q["FULL"] - q["READ_ONLY"]))
    read = 0 if read_main < -epsilon else 1
    write = 0 if write_main < -epsilon else 1
    return BITS_ACTION[(read, write)]


def derive_quantities(q: Mapping[str, float], epsilon: float) -> dict[str, Any]:
    advantages = {action: float(q[action] - q["FULL"]) for action in ACTIONS}
    suppression_action = max(SUPPRESSIONS, key=lambda action: (advantages[action], -TIE_PREFERENCE.index(action)))
    g = advantages[suppression_action]
    effects = {
        "read_w0": float(q["READ_ONLY"] - q["IGNORE"]),
        "read_w1": float(q["FULL"] - q["WRITE_ONLY"]),
        "write_r0": float(q["WRITE_ONLY"] - q["IGNORE"]),
        "write_r1": float(q["FULL"] - q["READ_ONLY"]),
        "interaction": float(q["FULL"] - q["READ_ONLY"] - q["WRITE_ONLY"] + q["IGNORE"]),
    }
    epsilon_preferred, ties = conservative_best(q, epsilon)
    best = exact_best(q)
    independent = independent_action(q, epsilon)
    if g > epsilon:
        region = "candidate_full_misaligned"
    elif g < -epsilon:
        region = "full_critical"
    else:
        region = "answer_silent_redundant"
    read_labels = (sign_label(effects["read_w0"], epsilon), sign_label(effects["read_w1"], epsilon))
    write_labels = (sign_label(effects["write_r0"], epsilon), sign_label(effects["write_r1"], epsilon))
    return {
        "advantages": advantages,
        "g": float(g),
        "best_suppression_action": suppression_action,
        "best_action": best,
        "epsilon_preferred_action": epsilon_preferred,
        "tie_actions": ties,
        "region": region,
        "effects": effects,
        "read_strict_sign_reversal": set(read_labels) == {"positive", "negative"},
        "write_strict_sign_reversal": set(write_labels) == {"positive", "negative"},
        "read_conditioning_label_change": read_labels[0] != read_labels[1],
        "write_conditioning_label_change": write_labels[0] != write_labels[1],
        "independent_action": independent,
        "independent_recovers_best": independent == best,
        "independent_within_epsilon_of_best": independent in ties,
    }


def weighted_answer_length(result: dict[str, Any]) -> float:
    weights = {row["answer"]: float(row["weight"]) for row in result["accepted_answers"]}
    return float(sum(weights[row["answer"]] * int(row["answer_token_length"]) for row in result["answer_tokenization"]))


def validate_and_flatten(results: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if len(results) != 400 or Counter(row["dataset"] for row in results) != Counter({"gqa": 200, "textvqa": 200}):
        raise ValueError("Frozen Stage B record count or dataset allocation changed")
    metadata = {row["id"]: row for row in manifest}
    if set(metadata) != {row["id"] for row in results}:
        raise ValueError("Results and candidate manifest do not match exactly")
    if len({str(row["selection_asset_key"]) for row in manifest}) != 400:
        raise ValueError("Frozen Stage B selection-asset uniqueness changed")
    q_rows: list[dict[str, Any]] = []
    advantage_rows: list[dict[str, Any]] = []
    maxima = {"full_sequence_parity": 0.0, "full_mean_parity": 0.0, "prestate": 0.0, "read_identity": 0.0, "write_identity": 0.0, "stored_effect": 0.0}
    for result in sorted(results, key=lambda row: row["id"]):
        source = metadata[result["id"]]
        if result["question"] != source["question"]:
            raise ValueError(f"Question mismatch for {result['id']}")
        if tuple(int(layer["layer"]) for layer in result["layers"]) != LAYERS:
            raise ValueError(f"Layer grid mismatch for {result['id']}")
        image_id = str(source["selection_asset_key"])
        question_id = str(source["sample_id"])
        answer_length = weighted_answer_length(result)
        for layer_record in result["layers"]:
            layer = int(layer_record["layer"])
            states = layer_record["states"]
            if set(states) != set(ACTIONS):
                raise ValueError(f"Incomplete state matrix for {result['id']} layer {layer}")
            full_correctness = float(states["FULL"]["official_correctness"])
            common: dict[str, Any] = {
                "sample_id": result["id"],
                "dataset": result["dataset"],
                "image_id": image_id,
                "question_id": question_id,
                "question": result["question"],
                "layer": layer,
                "full_official_correctness": full_correctness,
                "full_correct": full_correctness >= CORRECTNESS_THRESHOLD,
                "answer_length_weighted_tokens": answer_length,
                "prompt_token_length": int(result["prompt_token_length"]),
                "visual_token_first": int(result["visual_token_range"]["first"]),
                "visual_token_last": int(result["visual_token_range"]["last"]),
                "visual_token_count": int(result["visual_token_range"]["last"]) - int(result["visual_token_range"]["first"]) + 1,
            }
            q_row = dict(common)
            advantage_row = dict(common)
            for action in ACTIONS:
                state = states[action]
                maxima["prestate"] = max(maxima["prestate"], abs(float(state["prestate_injection_max_abs"])))
                maxima["read_identity"] = max(maxima["read_identity"], abs(float(state["read_hook_identity_max_abs"])))
                maxima["write_identity"] = max(maxima["write_identity"], abs(float(state["write_hook_identity_max_abs"])))
                q_row[f"correctness_{action.lower()}"] = float(state["official_correctness"])
                q_row[f"generated_answer_{action.lower()}"] = str(state["generated_answer"])
            for metric in METRICS:
                q = {action: float(states[action][f"{metric}_logprob"]) for action in ACTIONS}
                if not all(math.isfinite(value) for value in q.values()):
                    raise ValueError(f"Nonfinite Q for {result['id']} layer {layer}")
                for action, bits in ACTION_BITS.items():
                    q_row[f"q_{metric}_{bits[0]}{bits[1]}"] = q[action]
                derived = derive_quantities(q, EPSILON[metric])
                for action in ACTIONS:
                    advantage_row[f"a_{metric}_{action.lower()}"] = derived["advantages"][action]
                advantage_row[f"g_{metric}"] = derived["g"]
                advantage_row[f"best_suppression_action_{metric}"] = derived["best_suppression_action"]
                advantage_row[f"best_action_{metric}"] = derived["best_action"]
                advantage_row[f"epsilon_preferred_action_{metric}"] = derived["epsilon_preferred_action"]
                advantage_row[f"tie_actions_{metric}"] = "|".join(derived["tie_actions"])
                advantage_row[f"tie_size_{metric}"] = len(derived["tie_actions"])
                advantage_row[f"region_{metric}"] = derived["region"]
                advantage_row[f"q_range_{metric}"] = max(q.values()) - min(q.values())
                for effect, value in derived["effects"].items():
                    advantage_row[f"{effect}_{metric}"] = value
                    maxima["stored_effect"] = max(maxima["stored_effect"], abs(value - float(layer_record[f"{metric}_effects"][effect])))
                for field in ("read_strict_sign_reversal", "write_strict_sign_reversal", "read_conditioning_label_change", "write_conditioning_label_change", "independent_action", "independent_recovers_best", "independent_within_epsilon_of_best"):
                    advantage_row[f"{field}_{metric}"] = derived[field]
                baseline = float(result["baseline_full"][f"{metric}_logprob"])
                maxima[f"full_{metric}_parity"] = max(maxima[f"full_{metric}_parity"], abs(q["FULL"] - baseline))
            q_rows.append(q_row)
            advantage_rows.append(advantage_row)
    if maxima["full_sequence_parity"] != 0.0 or maxima["full_mean_parity"] != 0.0 or maxima["prestate"] != 0.0 or maxima["stored_effect"] != 0.0:
        raise ValueError(f"Frozen integrity failure: {maxima}")
    q_frame = pd.DataFrame(q_rows).sort_values(["dataset", "sample_id", "layer"]).reset_index(drop=True)
    advantage_frame = pd.DataFrame(advantage_rows).sort_values(["dataset", "sample_id", "layer"]).reset_index(drop=True)
    return q_frame, advantage_frame, maxima


def expanded_strata(frame: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    yield "all", frame
    yield "full_correct", frame[frame["full_correct"]]
    yield "full_wrong", frame[~frame["full_correct"]]


def build_summaries(q_frame: pd.DataFrame, advantage_frame: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    q_summary: list[dict[str, Any]] = []
    advantage_summary: list[dict[str, Any]] = []
    effect_summary: list[dict[str, Any]] = []
    best_frequency: list[dict[str, Any]] = []
    tie_summary: list[dict[str, Any]] = []
    region_summary: list[dict[str, Any]] = []
    for dataset in ("gqa", "textvqa"):
        for layer in LAYERS:
            q_part = q_frame[(q_frame.dataset == dataset) & (q_frame.layer == layer)]
            a_part = advantage_frame[(advantage_frame.dataset == dataset) & (advantage_frame.layer == layer)]
            for stratum, q_stratum in expanded_strata(q_part):
                a_stratum = a_part.loc[q_stratum.index]
                if q_stratum.empty:
                    continue
                for metric in METRICS:
                    for action, bits in ACTION_BITS.items():
                        column = f"q_{metric}_{bits[0]}{bits[1]}"
                        q_summary.append({"dataset": dataset, "layer": layer, "stratum": stratum, "metric": metric, "action": action, **summarize(q_stratum[column], ("q", dataset, layer, stratum, metric, action))})
                    for action in SUPPRESSIONS:
                        column = f"a_{metric}_{action.lower()}"
                        advantage_summary.append({"dataset": dataset, "layer": layer, "stratum": stratum, "metric": metric, "quantity": f"A_{action}", **summarize(a_stratum[column], ("adv", dataset, layer, stratum, metric, action))})
                    advantage_summary.append({"dataset": dataset, "layer": layer, "stratum": stratum, "metric": metric, "quantity": "G", **summarize(a_stratum[f"g_{metric}"], ("g", dataset, layer, stratum, metric))})
                    for effect in ("read_w0", "read_w1", "write_r0", "write_r1", "interaction"):
                        effect_summary.append({"dataset": dataset, "layer": layer, "stratum": stratum, "metric": metric, "effect": effect, **summarize(a_stratum[f"{effect}_{metric}"], ("effect", dataset, layer, stratum, metric, effect))})
            for metric in METRICS:
                counts = Counter(a_part[f"best_action_{metric}"])
                for action in ACTIONS:
                    best_frequency.append({"dataset": dataset, "layer": layer, "metric": metric, "action": action, "count": counts[action], "fraction": counts[action] / len(a_part)})
                epsilon = EPSILON[metric]
                tie_summary.append({
                    "dataset": dataset,
                    "layer": layer,
                    "metric": metric,
                    "n": len(a_part),
                    "multi_action_epsilon_tie_fraction": float((a_part[f"tie_size_{metric}"] > 1).mean()),
                    "all_four_numerically_tied_fraction": float((a_part[f"q_range_{metric}"] <= epsilon).mean()),
                    "best_suppression_silent_vs_full_fraction": float((a_part[f"g_{metric}"].abs() <= epsilon).mean()),
                    "best_suppression_within_0_05_fraction": float((a_part[f"g_mean"].abs() <= PRACTICAL_NEAR_TIE_MEAN).mean()) if metric == "mean" else math.nan,
                })
                region_counts = Counter(a_part[f"region_{metric}"])
                fractions = {name: region_counts[name] / len(a_part) for name in ("full_critical", "answer_silent_redundant", "candidate_full_misaligned")}
                modal_name, modal_fraction = max(fractions.items(), key=lambda item: item[1])
                region_summary.append({"dataset": dataset, "layer": layer, "metric": metric, **{f"{name}_fraction": value for name, value in fractions.items()}, "modal_region": modal_name if modal_fraction >= 0.5 else "heterogeneous", "modal_fraction": modal_fraction})
    return {"q": q_summary, "advantage": advantage_summary, "effect": effect_summary, "best": best_frequency, "tie": tie_summary, "region": region_summary}


def build_interaction_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in ("gqa", "textvqa"):
        for layer in LAYERS:
            part = frame[(frame.dataset == dataset) & (frame.layer == layer)]
            for metric in METRICS:
                interaction = summarize(part[f"interaction_{metric}"], ("interaction", dataset, layer, metric))
                rows.append({
                    "dataset": dataset,
                    "layer": layer,
                    "metric": metric,
                    **interaction,
                    "read_strict_sign_reversal_count": int(part[f"read_strict_sign_reversal_{metric}"].sum()),
                    "read_strict_sign_reversal_fraction": float(part[f"read_strict_sign_reversal_{metric}"].mean()),
                    "write_strict_sign_reversal_count": int(part[f"write_strict_sign_reversal_{metric}"].sum()),
                    "write_strict_sign_reversal_fraction": float(part[f"write_strict_sign_reversal_{metric}"].mean()),
                    "read_conditioning_label_change_fraction": float(part[f"read_conditioning_label_change_{metric}"].mean()),
                    "write_conditioning_label_change_fraction": float(part[f"write_conditioning_label_change_{metric}"].mean()),
                    "independent_main_effect_failure_count": int((~part[f"independent_recovers_best_{metric}"]).sum()),
                    "independent_main_effect_failure_fraction": float((~part[f"independent_recovers_best_{metric}"]).mean()),
                    "independent_main_effect_outside_epsilon_count": int((~part[f"independent_within_epsilon_of_best_{metric}"]).sum()),
                    "independent_main_effect_outside_epsilon_fraction": float((~part[f"independent_within_epsilon_of_best_{metric}"]).mean()),
                    "interaction_above_epsilon_count": int((part[f"interaction_{metric}"].abs() > EPSILON[metric]).sum()),
                    "interaction_above_epsilon_fraction": float((part[f"interaction_{metric}"].abs() > EPSILON[metric]).mean()),
                })
    return rows


def build_agreement(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    quantities = ["g", "read_w0", "read_w1", "write_r0", "write_r1", "interaction"]
    quantity_columns = [(quantity, f"{quantity}_mean", f"{quantity}_sequence") for quantity in quantities]
    quantity_columns.extend(
        (f"A_{action}", f"a_mean_{action.lower()}", f"a_sequence_{action.lower()}")
        for action in SUPPRESSIONS
    )
    for dataset in ("gqa", "textvqa"):
        for layer in LAYERS:
            part = frame[(frame.dataset == dataset) & (frame.layer == layer)]
            for quantity, mean_column, sequence_column in quantity_columns:
                mean_values = part[mean_column]
                sequence_values = part[sequence_column]
                rows.append({
                    "dataset": dataset,
                    "layer": layer,
                    "quantity": quantity,
                    "n": len(part),
                    "pearson_r": correlation(mean_values, sequence_values),
                    "spearman_r": correlation(mean_values, sequence_values, rank=True),
                    "sign_label_agreement": float(np.mean([sign_label(float(m), EPSILON["mean"]) == sign_label(float(s), EPSILON["sequence"]) for m, s in zip(mean_values, sequence_values)])),
                })
            rows.append({
                "dataset": dataset,
                "layer": layer,
                "quantity": "best_action",
                "n": len(part),
                "pearson_r": math.nan,
                "spearman_r": math.nan,
                "sign_label_agreement": float((part.best_action_mean == part.best_action_sequence).mean()),
            })
    return rows


def choose_mean_action(frame: pd.DataFrame, metric: str, epsilon: float) -> str:
    means = {action: float(frame[f"q_{metric}_{ACTION_BITS[action][0]}{ACTION_BITS[action][1]}"].mean()) for action in ACTIONS}
    return exact_best(means)


def cluster_summary(values: np.ndarray, sample_ids: np.ndarray, seed_parts: tuple[Any, ...]) -> dict[str, float | int]:
    per_sample: dict[str, list[float]] = defaultdict(list)
    for value, sample_id in zip(values, sample_ids):
        per_sample[str(sample_id)].append(float(value))
    sample_means = np.asarray([np.mean(per_sample[key]) for key in sorted(per_sample)], dtype=np.float64)
    return summarize(sample_means, seed_parts)


def build_fixed_policy_rows(q_frame: pd.DataFrame, advantage_frame: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    merged = q_frame.merge(advantage_frame[["sample_id", "layer"] + [f"best_action_{m}" for m in METRICS] + [f"tie_actions_{m}" for m in METRICS]], on=["sample_id", "layer"], validate="one_to_one")
    outputs: list[dict[str, Any]] = []
    maps: dict[str, Any] = {}
    for metric in METRICS:
        epsilon = EPSILON[metric]
        global_action = choose_mean_action(merged, metric, epsilon)
        per_layer = {str(layer): choose_mean_action(merged[merged.layer == layer], metric, epsilon) for layer in LAYERS}
        per_dataset_layer = {f"{dataset}:{layer}": choose_mean_action(merged[(merged.dataset == dataset) & (merged.layer == layer)], metric, epsilon) for dataset in ("gqa", "textvqa") for layer in LAYERS}
        maps[metric] = {"one_global_action": global_action, "per_layer": per_layer, "per_dataset_layer": per_dataset_layer}
        for policy in ("one_global_action", "one_action_per_layer", "one_action_per_dataset_layer", "always_full", "sample_layer_oracle"):
            selected: list[str] = []
            for row in merged.itertuples(index=False):
                if policy == "one_global_action":
                    action = global_action
                elif policy == "one_action_per_layer":
                    action = per_layer[str(row.layer)]
                elif policy == "one_action_per_dataset_layer":
                    action = per_dataset_layer[f"{row.dataset}:{row.layer}"]
                elif policy == "always_full":
                    action = "FULL"
                else:
                    action = getattr(row, f"best_action_{metric}")
                selected.append(action)
            selected_array = np.asarray(selected, dtype=object)
            for dataset_group in ("joint", "gqa", "textvqa"):
                mask = np.ones(len(merged), dtype=bool) if dataset_group == "joint" else merged.dataset.to_numpy() == dataset_group
                part = merged.loc[mask]
                actions = selected_array[mask]
                utility = np.asarray([float(part.iloc[i][f"q_{metric}_{ACTION_BITS[action][0]}{ACTION_BITS[action][1]}"]) for i, action in enumerate(actions)])
                full = part[f"q_{metric}_11"].to_numpy(dtype=float)
                oracle = np.asarray([max(float(part.iloc[i][f"q_{metric}_{bits[0]}{bits[1]}"]) for bits in ACTION_BITS.values()) for i in range(len(part))])
                relative = utility - full
                regret = oracle - utility
                exact_match = np.asarray([action == str(part.iloc[i][f"best_action_{metric}"]) for i, action in enumerate(actions)], dtype=float)
                epsilon_match = np.asarray([action in str(part.iloc[i][f"tie_actions_{metric}"]).split("|") for i, action in enumerate(actions)], dtype=float)
                full_correct = part.full_correct.to_numpy(dtype=bool)
                chosen_correctness = np.asarray([float(part.iloc[i][f"correctness_{action.lower()}"]) for i, action in enumerate(actions)])
                chosen_correct = chosen_correctness >= CORRECTNESS_THRESHOLD
                regressions = full_correct & ~chosen_correct
                improvements = ~full_correct & chosen_correct
                relative_summary = cluster_summary(relative, part.sample_id.to_numpy(), ("policy_relative", metric, policy, dataset_group))
                regret_summary = cluster_summary(regret, part.sample_id.to_numpy(), ("policy_regret", metric, policy, dataset_group))
                counts = Counter(actions)
                enabled_bits = np.asarray([sum(ACTION_BITS[action]) for action in actions], dtype=float)
                outputs.append({
                    "metric": metric,
                    "policy": policy,
                    "dataset_group": dataset_group,
                    "n_sample_layer_pairs": len(part),
                    "n_samples": int(part.sample_id.nunique()),
                    "mean_utility_relative_full": relative_summary["mean"],
                    "utility_relative_full_ci_low": relative_summary["mean_ci_low"],
                    "utility_relative_full_ci_high": relative_summary["mean_ci_high"],
                    "mean_regret_to_oracle": regret_summary["mean"],
                    "regret_ci_low": regret_summary["mean_ci_low"],
                    "regret_ci_high": regret_summary["mean_ci_high"],
                    "oracle_exact_match_fraction": float(exact_match.mean()),
                    "oracle_epsilon_tie_set_match_fraction": float(epsilon_match.mean()),
                    "full_correct_regression_pairs": int(regressions.sum()),
                    "full_correct_regression_unique_samples": int(part.loc[regressions, "sample_id"].nunique()),
                    "full_wrong_improvement_pairs": int(improvements.sum()),
                    "full_wrong_improvement_unique_samples": int(part.loc[improvements, "sample_id"].nunique()),
                    "selected_ignore": counts["IGNORE"],
                    "selected_read_only": counts["READ_ONLY"],
                    "selected_write_only": counts["WRITE_ONLY"],
                    "selected_full": counts["FULL"],
                    "mean_enabled_action_bits_descriptive": float(enabled_bits.mean()),
                    "selection_note": "in-sample descriptive; enabled-bit count is not FLOPs or an acceleration estimate",
                })
    return outputs, maps


def distribution_from_image_ids(rows: Iterable[tuple[str, str]]) -> dict[str, Any]:
    groups: dict[str, set[str]] = defaultdict(set)
    for image_id, question_id in rows:
        groups[str(image_id)].add(str(question_id))
    counts = np.asarray([len(values) for values in groups.values()], dtype=int)
    histogram = Counter(int(value) for value in counts)
    return {
        "record_count": int(counts.sum()) if counts.size else 0,
        "image_count": int(counts.size),
        "images_with_at_least_two_questions": int((counts >= 2).sum()) if counts.size else 0,
        "records_in_multi_question_images": int(counts[counts >= 2].sum()) if counts.size else 0,
        "questions_per_image_histogram": {str(key): int(value) for key, value in sorted(histogram.items())},
        "questions_per_image_quantiles": {name: float(np.quantile(counts, q)) if counts.size else math.nan for name, q in (("q00", 0), ("q25", 0.25), ("q50", 0.5), ("q75", 0.75), ("q90", 0.9), ("q95", 0.95), ("q99", 0.99), ("q100", 1.0))},
    }


def arrow_rows(paths: list[Path], columns: tuple[str, ...]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("rb") as handle:
            reader = ipc.open_stream(handle)
            for batch in reader:
                selected = pa.RecordBatch.from_arrays([batch.column(batch.schema.get_field_index(column)) for column in columns], names=list(columns))
                yield from selected.to_pylist()


def same_image_feasibility(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    selected_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        selected_by_dataset[row["benchmark"]].append(row)
    pool_root = Path("/data/dataset/dynamic_mllm/Qwen2.5VL/easy_hard_5k/complete_correct_wrong_pools_20260713")
    pool_paths = {
        "gqa": [pool_root / "gqa_complete_correct_2000.jsonl", pool_root / "gqa_complete_wrong_2000.jsonl"],
        "textvqa": [pool_root / "textvqa_complete_correct_1000.jsonl", pool_root / "textvqa_complete_wrong_1000.jsonl"],
    }
    discovery_pool: dict[str, Any] = {}
    for dataset, paths in pool_paths.items():
        rows = [row for path in paths for row in read_jsonl(path)]
        selected_images = {str(row["selection_asset_key"]) for row in selected_by_dataset[dataset]}
        all_pairs = [
            (
                str(row.get("source_asset_id") or (pool_root / "images" / dataset / Path(str(row["image_path"])).name)),
                str(row["sample_id"]),
            )
            for row in rows
        ]
        remaining_pairs = [(image, question) for image, question in all_pairs if image not in selected_images]
        discovery_pool[dataset] = {
            "source": "easy_hard_5k train-derived correctness-selected metadata",
            "all": distribution_from_image_ids(all_pairs),
            "after_stage_b_image_exclusion": distribution_from_image_ids(remaining_pairs),
            "confirmation_eligible": False,
        }

    textvqa_root = Path("/data/dataset/huggingface/datasets/lmms-lab___textvqa/default/0.0.0/9c0699cd19768ac5ab97568f6b3cbac4c0062884")
    textvqa_paths = sorted(textvqa_root.glob("textvqa-validation-*.arrow"))
    audit = json.loads(Path("outputs/stage_c/manifest/stage_c_eligibility_overlap_audit_v1.json").read_text(encoding="utf-8"))
    invalid_question_ids = {str(row["question_id"]) for row in audit["invalid_records"]}
    stage_c_rows = read_jsonl(Path("outputs/stage_c/manifest/stage_c_manifest_v1.jsonl"))
    inspected_textvqa_images = {str(row["image_id"]) for row in stage_c_rows}
    stage_b_textvqa_images = {
        str(row["source_asset_id"]).split(":", 1)[-1]
        for row in selected_by_dataset["textvqa"]
        if row.get("source_asset_id")
    }
    unresolved_stage_b_textvqa_images = sum(not row.get("source_asset_id") for row in selected_by_dataset["textvqa"])
    textvqa_valid_pairs: list[tuple[str, str]] = []
    for row in arrow_rows(textvqa_paths, ("image_id", "question_id", "question", "answers")):
        qid = str(row["question_id"])
        if qid in invalid_question_ids or not str(row["question"] or "").strip() or not row["answers"]:
            continue
        textvqa_valid_pairs.append((str(row["image_id"]), qid))
    textvqa_future_pairs = [(image, question) for image, question in textvqa_valid_pairs if image not in inspected_textvqa_images and image not in stage_b_textvqa_images]

    gqa_path = Path("/data/dataset/huggingface/datasets/lmms-lab___gqa/val_balanced_instructions/0.0.0/a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8/gqa-val.arrow")
    stage_b_gqa_images = {str(row["source_asset_id"]).split(":", 1)[-1] for row in selected_by_dataset["gqa"]}
    gqa_pairs: list[tuple[str, str]] = []
    for row in arrow_rows([gqa_path], ("id", "imageId", "question", "answer")):
        if str(row["id"] or "").strip() and str(row["imageId"] or "").strip() and str(row["question"] or "").strip() and str(row["answer"] or "").strip():
            gqa_pairs.append((str(row["imageId"]), str(row["id"])))
    gqa_future_pairs = [(image, question) for image, question in gqa_pairs if image not in stage_b_gqa_images]

    stage_b_q = {
        dataset: distribution_from_image_ids((str(row["selection_asset_key"]), str(row["sample_id"])) for row in selected_by_dataset[dataset])
        for dataset in ("gqa", "textvqa")
    }
    return {
        "schema_version": "v3_same_image_feasibility_v1",
        "metadata_only": True,
        "architecture": {
            "visual_tokens_precede_question": True,
            "causal_mask_blocks_visual_queries_from_later_question_tokens": True,
            "same_image_and_same_preceding_prefix_implies_query_invariant_visual_prefix_and_write": True,
            "evidence": "outputs/stage_a/architecture_causal_graph.md; visual_future_attention_mass_max=0",
            "numerical_cross_question_sanity_executed": False,
            "interpretation": "structurally exact under the pinned token order/mask; numerical cross-question equality remains a preflight sanity check",
        },
        "stage_b_with_four_action_q": {
            "gqa": stage_b_q["gqa"],
            "textvqa": stage_b_q["textvqa"],
            "same_image_groups_with_q_at_common_layers": 0,
            "discovery_same_image_q_summary_available": False,
            "reason": "the frozen Stage B manifest intentionally contains 400 unique effective images",
        },
        "unscored_discovery_metadata": discovery_pool,
        "future_confirmation_metadata": {
            "gqa": {
                "source": "lmms-lab/GQA val_balanced_instructions revision a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8",
                "validity_level": "metadata-eligible only; processor/token checks remain required",
                "all_metadata_eligible": distribution_from_image_ids(gqa_pairs),
                "new_after_all_inspected_gqa_image_exclusions": distribution_from_image_ids(gqa_future_pairs),
                "stage_b_image_overlap_removed": len(gqa_pairs) - len(gqa_future_pairs),
            },
            "textvqa": {
                "source": "lmms-lab/textvqa validation revision 9c0699cd19768ac5ab97568f6b3cbac4c0062884",
                "validity_level": "reuses frozen Stage C technical-invalid audit rules",
                "all_technically_valid": distribution_from_image_ids(textvqa_valid_pairs),
                "new_after_stage_b_and_stage_c_image_exclusions": distribution_from_image_ids(textvqa_future_pairs),
                "excluded_inspected_stage_c_image_count": len(inspected_textvqa_images),
                "stage_b_official_image_overlap_removed": len(textvqa_valid_pairs) - len([pair for pair in textvqa_valid_pairs if pair[0] not in stage_b_textvqa_images]),
                "stage_b_records_without_official_image_id": unresolved_stage_b_textvqa_images,
                "invalid_question_count": len(invalid_question_ids),
            },
        },
    }


def textvqa_layer0_reinterpretation(q_frame: pd.DataFrame, advantages: pd.DataFrame) -> dict[str, Any]:
    q = q_frame[(q_frame.dataset == "textvqa") & (q_frame.layer == 0)].reset_index(drop=True)
    a = advantages[(advantages.dataset == "textvqa") & (advantages.layer == 0)].reset_index(drop=True)
    counts = Counter(a.best_action_mean)
    write_winners = a.best_action_mean == "WRITE_ONLY"
    ignore_close = (q.q_mean_00 - q.q_mean_01).abs() <= PRACTICAL_NEAR_TIE_MEAN
    read_only_close = (q.q_mean_10 - q.q_mean_01).abs() <= PRACTICAL_NEAR_TIE_MEAN
    return {
        "n": len(q),
        "best_action_counts_per_token": {action: counts[action] for action in ACTIONS},
        "write_only_best_count": int(write_winners.sum()),
        "among_write_only_best_ignore_within_0_05_count": int((write_winners & ignore_close).sum()),
        "among_write_only_best_read_only_within_0_05_count": int((write_winners & read_only_close).sum()),
        "among_write_only_best_either_other_suppression_within_0_05_count": int((write_winners & (ignore_close | read_only_close)).sum()),
        "mean_full_minus_write_only": float((q.q_mean_11 - q.q_mean_01).mean()),
        "median_full_minus_write_only": float((q.q_mean_11 - q.q_mean_01).median()),
        "mean_interaction": float(a.interaction_mean.mean()),
        "old_stage_c_context": "Outcome B preserved; the held-out narrow contrast failed both structured-null superiority gates",
    }


def main() -> int:
    args = parse_args()
    results_path = Path(args.results)
    candidate_path = Path(args.candidate_manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = read_jsonl(results_path)
    manifest = read_jsonl(candidate_path)
    q_frame, advantage_frame, maxima = validate_and_flatten(results, manifest)

    q_path = output_dir / "four_action_q_v1.parquet"
    advantage_path = output_dir / "full_relative_advantages_v1.parquet"
    q_frame.to_parquet(q_path, index=False, compression="zstd")
    advantage_frame.to_parquet(advantage_path, index=False, compression="zstd")

    summaries = build_summaries(q_frame, advantage_frame)
    interaction_rows = build_interaction_summary(advantage_frame)
    agreement_rows = build_agreement(advantage_frame)
    policy_rows, policy_maps = build_fixed_policy_rows(q_frame, advantage_frame)
    feasibility = same_image_feasibility(manifest)

    output_tables = {
        "layer_q_summary_v1.csv": summaries["q"],
        "layer_advantage_summary_v1.csv": summaries["advantage"],
        "conditional_effect_summary_v1.csv": summaries["effect"],
        "best_action_frequencies_v1.csv": summaries["best"],
        "tie_summary_v1.csv": summaries["tie"],
        "region_summary_v1.csv": summaries["region"],
        "interaction_summary_v1.csv": interaction_rows,
        "sequence_per_token_agreement_v1.csv": agreement_rows,
        "fixed_policy_regret_v1.csv": policy_rows,
    }
    for filename, rows in output_tables.items():
        write_csv(output_dir / filename, rows)
    write_json(output_dir / "same_image_feasibility_v1.json", feasibility)
    write_json(output_dir / "fixed_policy_maps_v1.json", policy_maps)
    write_json(output_dir / "textvqa_layer0_reinterpretation_v1.json", textvqa_layer0_reinterpretation(q_frame, advantage_frame))

    examples: list[dict[str, Any]] = []
    for dataset in ("gqa", "textvqa"):
        part = advantage_frame[advantage_frame.dataset == dataset].copy()
        part["abs_interaction"] = part.interaction_mean.abs()
        for row in part.sort_values(["abs_interaction", "sample_id", "layer"], ascending=[False, True, True]).head(5).to_dict("records"):
            examples.append({key: row[key] for key in ("sample_id", "dataset", "image_id", "question_id", "question", "layer", "best_action_mean", "g_mean", "read_w0_mean", "read_w1_mean", "write_r0_mean", "write_r1_mean", "interaction_mean")})
    write_json(output_dir / "interaction_examples_v1.json", {"selection": "top five absolute per-token interactions per dataset after aggregate computation; discovery illustration only", "examples": examples})

    required_outputs = [q_path, advantage_path, output_dir / "fixed_policy_regret_v1.csv", output_dir / "interaction_summary_v1.csv", output_dir / "same_image_feasibility_v1.json"]
    required_outputs.extend(output_dir / name for name in output_tables if name not in {"fixed_policy_regret_v1.csv", "interaction_summary_v1.csv"})
    required_outputs.extend([output_dir / "fixed_policy_maps_v1.json", output_dir / "textvqa_layer0_reinterpretation_v1.json", output_dir / "interaction_examples_v1.json"])
    analysis_manifest = {
        "schema_version": "v3_stage_b_reanalysis_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "analysis_kind": "deterministic reanalysis of inspected discovery outcomes",
        "model_loaded": False,
        "new_intervention_outcomes_collected": False,
        "training_executed": False,
        "v2_stage_c_decision_preserved": "Outcome B",
        "inputs": {
            str(results_path): sha256(results_path),
            str(candidate_path): sha256(candidate_path),
            "plans/dynamic_mllm_read_write_policy_conditional_plan_v3.md": sha256(Path("plans/dynamic_mllm_read_write_policy_conditional_plan_v3.md")),
            "reports/v3_migration_audit.md": sha256(Path("reports/v3_migration_audit.md")),
            "outputs/stage_a/stage_a_summary.json": sha256(Path("outputs/stage_a/stage_a_summary.json")),
            "reports/stage_c_conclusion.md": sha256(Path("reports/stage_c_conclusion.md")),
        },
        "counts": {"records": 400, "sample_layer_pairs": len(q_frame), "action_cells": len(q_frame) * 4, "datasets": {"gqa": 200, "textvqa": 200}, "layers": list(LAYERS)},
        "mapping": {action: {"read": bits[0], "write": bits[1]} for action, bits in ACTION_BITS.items()},
        "primary_cross_sample_metric": "per-token accepted-reference log-likelihood",
        "secondary_metric": "sequence accepted-reference log-likelihood",
        "epsilon": EPSILON,
        "practical_near_tie_mean_secondary": PRACTICAL_NEAR_TIE_MEAN,
        "best_action_rule": "exact argmax with deterministic exact-tie preference FULL > READ_ONLY > WRITE_ONLY > IGNORE",
        "tie_rule": "actions within metric epsilon of the maximum are additionally recorded as a numerical near-tie set",
        "independent_main_effect_rule": "choose each bit from the average of its two conditional effects; prefer bit ON within epsilon",
        "bootstrap": {"unit": "sample", "replicates": BOOTSTRAP_REPLICATES, "base_seed": BOOTSTRAP_SEED, "interval": "percentile 95%", "policy_aggregation": "resample sample-level means across all eight layers"},
        "fixed_policy_note": "actions are selected and described on the same inspected discovery data; results are optimistic descriptive fits, not held-out policy evaluation",
        "model": {
            "id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
            "runtime_metadata": "outputs/stage_b/runtime.json",
        },
        "scoring_version": "frozen Stage B accepted-reference likelihood scoring",
        "code_commit": "unavailable_not_a_git_repository",
        "integrity": {"same_image_question_from_manifest": True, "identical_dense_prestate_by_implementation_and_recorded_injection": maxima["prestate"] == 0.0, "single_layer_intervention_and_unchanged_dense_suffix_verified_by_migration_audit": True, "identical_scoring": True, "full_sequence_parity_max_abs": maxima["full_sequence_parity"], "full_mean_parity_max_abs": maxima["full_mean_parity"], "read_identity_max_abs": maxima["read_identity"], "write_identity_max_abs": maxima["write_identity"], "stored_effect_recompute_max_abs": maxima["stored_effect"], "missing_or_imputed_values": False},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pa.__version__},
        "artifacts": {str(path): sha256(path) for path in sorted(set(required_outputs))},
    }
    write_json(output_dir / "analysis_manifest.json", analysis_manifest)
    print(json.dumps({"status": "complete", "sample_layer_pairs": len(q_frame), "artifacts": len(analysis_manifest["artifacts"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
