from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


STATES = ("IGNORE", "READ_ONLY", "WRITE_ONLY", "FULL")
EFFECTS = ("read_w0", "read_w1", "write_r0", "write_r1", "interaction")


def effect_label(value: float, epsilon: float) -> str:
    if value > epsilon:
        return "positive"
    if value < -epsilon:
        return "negative"
    return "answer_silent"


def is_correct(score: float, threshold: float) -> bool:
    return float(score) >= threshold


def behavior_category(
    full_score: float,
    intervention_score: float,
    full_answer: str,
    intervention_answer: str,
    threshold: float,
) -> str:
    full_correct = is_correct(full_score, threshold)
    intervention_correct = is_correct(intervention_score, threshold)
    if not full_correct and intervention_correct:
        return "full_wrong_to_intervention_correct"
    if full_correct and intervention_correct:
        return "full_correct_to_intervention_correct"
    if full_correct and not intervention_correct:
        return "full_correct_to_intervention_wrong"
    if full_answer == intervention_answer:
        return "unchanged_wrong"
    return "changed_but_still_wrong"


def summarize_values(
    values: Iterable[float], bootstrap_replicates: int, seed: int
) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        raise ValueError("Cannot summarize an empty value collection")
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, array.size, size=(bootstrap_replicates, array.size))
    boot_means = array[draws].mean(axis=1)
    return {
        "n_samples": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "mean_ci_low": float(np.quantile(boot_means, 0.025)),
        "mean_ci_high": float(np.quantile(boot_means, 0.975)),
    }


def stable_seed(base_seed: int, parts: Iterable[Any]) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return (base_seed + digest) % (2**32)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def dataset_groups(dataset: str) -> tuple[str, str]:
    return dataset, "joint"


def weighted_answer_length(result: dict[str, Any]) -> float:
    weights = {row["answer"]: float(row["weight"]) for row in result["accepted_answers"]}
    return float(
        sum(weights[row["answer"]] * int(row["answer_token_length"])
            for row in result["answer_tokenization"])
    )


def answer_length_bin(length: float) -> str:
    if length <= 1.0:
        return "1"
    if length <= 3.0:
        return "2-3"
    return "4+"


def flatten_results(
    results: list[dict[str, Any]],
    epsilon_sequence: float,
    epsilon_mean: float,
    correctness_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    state_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    for result in results:
        length = weighted_answer_length(result)
        for layer in result["layers"]:
            layer_index = int(layer["layer"])
            full = layer["states"]["FULL"]
            full_correct = is_correct(full["official_correctness"], correctness_threshold)
            common = {
                "id": result["id"],
                "dataset": result["dataset"],
                "layer": layer_index,
                "full_correct": full_correct,
                "inherited_bucket": result["inherited_bucket"],
                "answer_length": length,
                "answer_length_bin": answer_length_bin(length),
            }
            for state_name in STATES:
                state = layer["states"][state_name]
                for score_type in ("sequence", "mean"):
                    state_rows.append(
                        {
                            **common,
                            "score_type": score_type,
                            "state": state_name,
                            "value": float(state[f"{score_type}_logprob"]),
                        }
                    )
            for score_type, epsilon in (
                ("sequence", epsilon_sequence),
                ("mean", epsilon_mean),
            ):
                for effect_name, value in layer[f"{score_type}_effects"].items():
                    effect_rows.append(
                        {
                            **common,
                            "score_type": score_type,
                            "effect": effect_name,
                            "value": float(value),
                            "effect_label": effect_label(float(value), epsilon),
                        }
                    )
            for state_name in ("IGNORE", "READ_ONLY", "WRITE_ONLY"):
                state = layer["states"][state_name]
                delta_sequence = float(state["sequence_logprob"] - full["sequence_logprob"])
                delta_mean = float(state["mean_logprob"] - full["mean_logprob"])
                category = behavior_category(
                    full["official_correctness"],
                    state["official_correctness"],
                    full["generated_answer"],
                    state["generated_answer"],
                    correctness_threshold,
                )
                branch_rows.append(
                    {
                        **common,
                        "intervention_state": state_name,
                        "delta_sequence_vs_full": delta_sequence,
                        "delta_mean_vs_full": delta_mean,
                        "likelihood_direction": effect_label(delta_sequence, epsilon_sequence),
                        "behavior_category": category,
                        "full_generated_answer": full["generated_answer"],
                        "intervention_generated_answer": state["generated_answer"],
                        "full_official_correctness": float(full["official_correctness"]),
                        "intervention_official_correctness": float(state["official_correctness"]),
                    }
                )
    return state_rows, effect_rows, branch_rows


def grouped_summaries(
    rows: list[dict[str, Any]],
    group_fields: tuple[str, ...],
    value_field: str,
    bootstrap_replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        for dataset_group in dataset_groups(str(row["dataset"])):
            expanded = {**row, "dataset_group": dataset_group}
            key = tuple(expanded[field] for field in group_fields)
            groups[key].append(float(row[value_field]))
    output = []
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        summary = summarize_values(
            groups[key], bootstrap_replicates, stable_seed(seed, key)
        )
        output.append({**dict(zip(group_fields, key)), **summary})
    return output


def threshold_fraction_rows(effect_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], Counter] = defaultdict(Counter)
    for row in effect_rows:
        for dataset_group in dataset_groups(str(row["dataset"])):
            key = (
                dataset_group,
                row["layer"],
                row["full_correct"],
                row["score_type"],
                row["effect"],
            )
            groups[key][row["effect_label"]] += 1
    output = []
    for key, counts in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        total = sum(counts.values())
        output.append(
            {
                "dataset_group": key[0],
                "layer": key[1],
                "full_correct": key[2],
                "score_type": key[3],
                "effect": key[4],
                "n_samples": total,
                "positive_fraction": counts["positive"] / total,
                "negative_fraction": counts["negative"] / total,
                "answer_silent_fraction": counts["answer_silent"] / total,
            }
        )
    return output


def behavior_count_rows(branch_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], Counter] = defaultdict(Counter)
    for row in branch_rows:
        for dataset_group in dataset_groups(str(row["dataset"])):
            key = (dataset_group, row["layer"], row["intervention_state"])
            groups[key][row["behavior_category"]] += 1
    output = []
    for key, counts in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        for category, count in sorted(counts.items()):
            output.append(
                {
                    "dataset_group": key[0],
                    "layer": key[1],
                    "intervention_state": key[2],
                    "behavior_category": category,
                    "count": count,
                }
            )
    return output


def likelihood_behavior_rows(branch_rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], Counter] = defaultdict(Counter)
    for row in branch_rows:
        intervention_correct = is_correct(row["intervention_official_correctness"], threshold)
        full_correct = is_correct(row["full_official_correctness"], threshold)
        if row["likelihood_direction"] == "positive" and not intervention_correct:
            relation = "likelihood_improved_but_still_wrong"
        elif row["likelihood_direction"] == "positive" and not full_correct and intervention_correct:
            relation = "likelihood_improved_and_corrected"
        elif row["likelihood_direction"] == "negative" and full_correct and not intervention_correct:
            relation = "likelihood_decreased_and_regressed"
        else:
            relation = "other"
        for dataset_group in dataset_groups(str(row["dataset"])):
            groups[(dataset_group, row["layer"], row["intervention_state"])][relation] += 1
    output = []
    for key, counts in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        output.append(
            {
                "dataset_group": key[0],
                "layer": key[1],
                "intervention_state": key[2],
                **{name: counts[name] for name in (
                    "likelihood_improved_but_still_wrong",
                    "likelihood_improved_and_corrected",
                    "likelihood_decreased_and_regressed",
                    "other",
                )},
            }
        )
    return output


def sample_layer_averages(effect_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    exemplars: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in effect_rows:
        key = (row["id"], row["dataset"], row["full_correct"], row["score_type"], row["effect"])
        groups[key].append(float(row["value"]))
        exemplars[key] = row
    return [
        {
            "id": key[0],
            "dataset": key[1],
            "full_correct": key[2],
            "score_type": key[3],
            "effect": key[4],
            "answer_length": exemplars[key]["answer_length"],
            "answer_length_bin": exemplars[key]["answer_length_bin"],
            "value": float(np.mean(values)),
        }
        for key, values in groups.items()
    ]


def agreement_rows(effect_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paired: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in effect_rows:
        key = (row["id"], row["dataset"], row["layer"], row["effect"])
        paired[key][row["score_type"]] = row
    groups: dict[tuple[Any, ...], list[tuple[float, float, bool]]] = defaultdict(list)
    for (_, dataset, layer, effect), pair in paired.items():
        if set(pair) != {"sequence", "mean"}:
            raise ValueError("Sequence and mean effect rows do not pair exactly")
        item = (
            float(pair["sequence"]["value"]),
            float(pair["mean"]["value"]),
            pair["sequence"]["effect_label"] == pair["mean"]["effect_label"],
        )
        for dataset_group in dataset_groups(dataset):
            groups[(dataset_group, layer, effect)].append(item)
    output = []
    for key, values in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        sequence = np.asarray([row[0] for row in values], dtype=np.float64)
        mean = np.asarray([row[1] for row in values], dtype=np.float64)
        correlation = float(np.corrcoef(sequence, mean)[0, 1]) if sequence.std() and mean.std() else math.nan
        output.append(
            {
                "dataset_group": key[0],
                "layer": key[1],
                "effect": key[2],
                "n_samples": len(values),
                "pearson_r": correlation,
                "threshold_label_agreement": float(np.mean([row[2] for row in values])),
            }
        )
    return output


def answer_length_correlation_rows(sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[tuple[float, float]]] = defaultdict(list)
    for row in sample_rows:
        for dataset_group in dataset_groups(str(row["dataset"])):
            groups[(dataset_group, row["score_type"], row["effect"])].append(
                (float(row["answer_length"]), float(row["value"]))
            )
    output = []
    for key, values in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        lengths = np.asarray([item[0] for item in values], dtype=np.float64)
        effects = np.asarray([item[1] for item in values], dtype=np.float64)
        correlation = float(np.corrcoef(lengths, effects)[0, 1]) if lengths.std() and effects.std() else math.nan
        output.append(
            {
                "dataset_group": key[0],
                "score_type": key[1],
                "effect": key[2],
                "n_samples": len(values),
                "pearson_r_answer_length_vs_layer_mean_effect": correlation,
            }
        )
    return output


def line_svg(
    path: Path,
    rows: list[dict[str, Any]],
    series_field: str,
    value_field: str,
    title: str,
    y_label: str,
) -> None:
    width, height = 900, 520
    left, right, top, bottom = 85, 25, 45, 70
    layers = sorted({int(row["layer"]) for row in rows})
    series = sorted({str(row[series_field]) for row in rows})
    lookup = {(str(row[series_field]), int(row["layer"])): float(row[value_field]) for row in rows}
    values = [value for value in lookup.values() if math.isfinite(value)]
    if not layers or not values:
        return
    y_min, y_max = min(values), max(values)
    if y_min <= 0 <= y_max:
        margin = max((y_max - y_min) * 0.08, 1e-12)
        y_min -= margin
        y_max += margin
    elif y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    plot_w, plot_h = width - left - right, height - top - bottom
    x = lambda layer: left + (layer - layers[0]) / max(layers[-1] - layers[0], 1) * plot_w
    y = lambda value: top + (y_max - value) / (y_max - y_min) * plot_h
    colors = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22")
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>']
    svg.append(f'<text x="{width/2}" y="25" text-anchor="middle" font-size="18">{title}</text>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="black"/>')
    svg.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="black"/>')
    if y_min <= 0 <= y_max:
        svg.append(f'<line x1="{left}" y1="{y(0)}" x2="{left+plot_w}" y2="{y(0)}" stroke="#999" stroke-dasharray="4 4"/>')
    for layer in layers:
        svg.append(f'<text x="{x(layer)}" y="{top+plot_h+22}" text-anchor="middle" font-size="12">{layer}</text>')
    for tick in np.linspace(y_min, y_max, 5):
        svg.append(f'<text x="{left-8}" y="{y(float(tick))+4}" text-anchor="end" font-size="11">{tick:.3g}</text>')
    for index, name in enumerate(series):
        color = colors[index % len(colors)]
        points = [(x(layer), y(lookup[(name, layer)])) for layer in layers if (name, layer) in lookup]
        svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="' + " ".join(f"{px:.1f},{py:.1f}" for px, py in points) + '"/>')
        for px, py in points:
            svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{color}"/>')
        legend_y = top + 16 * index
        svg.append(f'<line x1="{left+plot_w-175}" y1="{legend_y}" x2="{left+plot_w-155}" y2="{legend_y}" stroke="{color}" stroke-width="2"/>')
        svg.append(f'<text x="{left+plot_w-150}" y="{legend_y+4}" font-size="11">{name}</text>')
    svg.append(f'<text x="{left+plot_w/2}" y="{height-18}" text-anchor="middle" font-size="13">layer</text>')
    svg.append(f'<text x="18" y="{top+plot_h/2}" text-anchor="middle" font-size="13" transform="rotate(-90 18 {top+plot_h/2})">{y_label}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def create_plots(
    output_dir: Path,
    state_summary: list[dict[str, Any]],
    effect_summary: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    behavior_counts: list[dict[str, Any]],
    likelihood_behavior: list[dict[str, Any]],
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for dataset in ("gqa", "textvqa", "joint"):
        states = [row for row in state_summary if row["dataset_group"] == dataset and row["score_type"] == "sequence" and row["full_correct"] == "all"]
        line_svg(plot_dir / f"state_scores_{dataset}.svg", states, "state", "mean", f"{dataset}: reference sequence scores", "mean log-likelihood")
        effects = [row for row in effect_summary if row["dataset_group"] == dataset and row["score_type"] == "sequence" and row["full_correct"] == "all"]
        line_svg(plot_dir / f"signed_effects_{dataset}.svg", effects, "effect", "mean", f"{dataset}: signed READ/WRITE effects", "mean score delta")
        thresholds = [row for row in threshold_rows if row["dataset_group"] == dataset and row["score_type"] == "sequence" and row["full_correct"] == "all"]
        expanded = []
        for row in thresholds:
            expanded.extend([
                {"layer": row["layer"], "series": f"{row['effect']}:positive", "fraction": row["positive_fraction"]},
                {"layer": row["layer"], "series": f"{row['effect']}:negative", "fraction": row["negative_fraction"]},
            ])
        line_svg(plot_dir / f"threshold_fractions_{dataset}.svg", expanded, "series", "fraction", f"{dataset}: fractions beyond frozen epsilon", "sample fraction")
        behavior_totals: dict[tuple[int, str], int] = defaultdict(int)
        for row in behavior_counts:
            if row["dataset_group"] == dataset:
                behavior_totals[(int(row["layer"]), str(row["behavior_category"]))] += int(row["count"])
        behavior_plot = [
            {"layer": layer, "category": category, "count": count}
            for (layer, category), count in behavior_totals.items()
        ]
        line_svg(plot_dir / f"greedy_behavior_{dataset}.svg", behavior_plot, "category", "count", f"{dataset}: greedy behavior across three ablations", "sample-branch count")
        relation_plot = []
        for row in likelihood_behavior:
            if row["dataset_group"] != dataset:
                continue
            for relation in (
                "likelihood_improved_but_still_wrong",
                "likelihood_improved_and_corrected",
                "likelihood_decreased_and_regressed",
            ):
                relation_plot.append(
                    {"layer": row["layer"], "relation": relation, "count": row[relation]}
                )
        relation_totals: dict[tuple[int, str], int] = defaultdict(int)
        for row in relation_plot:
            relation_totals[(int(row["layer"]), str(row["relation"]))] += int(row["count"])
        line_svg(
            plot_dir / f"likelihood_behavior_relation_{dataset}.svg",
            [
                {"layer": layer, "relation": relation, "count": count}
                for (layer, relation), count in relation_totals.items()
            ],
            "relation",
            "count",
            f"{dataset}: likelihood and greedy outcome relation",
            "sample-branch count",
        )


def add_full_correct_strata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded = []
    for row in rows:
        expanded.append({**row, "full_correct": "all"})
        expanded.append({**row, "full_correct": str(bool(row["full_correct"])).lower()})
    return expanded


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate Stage B reference-likelihood results.")
    parser.add_argument("--config", default="configs/stage_b.yaml")
    parser.add_argument("--results", default="outputs/stage_b/stage_b_results_v1.jsonl")
    parser.add_argument("--output-dir", default="outputs/stage_b/analysis_v1")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    results = read_jsonl(Path(args.results))
    result_path = Path(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epsilon_sequence = float(config["epsilon_sequence"])
    epsilon_mean = float(config["epsilon_mean"])
    correctness_threshold = float(config["behavior_correctness_threshold"])
    bootstrap_replicates = int(config["bootstrap_replicates"])
    seed = int(config["seed"])

    state_rows, effect_rows, branch_rows = flatten_results(
        results, epsilon_sequence, epsilon_mean, correctness_threshold
    )
    relabel_rows = []
    for result in results:
        inherited_correct = result["inherited_bucket"] == "complete_correct"
        pinned_score = float(result["baseline_full"]["official_correctness"])
        pinned_correct = is_correct(pinned_score, correctness_threshold)
        relabel_rows.append(
            {
                "id": result["id"],
                "dataset": result["dataset"],
                "inherited_bucket": result["inherited_bucket"],
                "inherited_correct": inherited_correct,
                "pinned_full_official_score": pinned_score,
                "pinned_full_correct": pinned_correct,
                "label_agreement": inherited_correct == pinned_correct,
                "pinned_full_generated_answer": result["baseline_full"]["generated_answer"],
            }
        )
    state_stratified = add_full_correct_strata(state_rows)
    effect_stratified = add_full_correct_strata(effect_rows)
    state_summary = grouped_summaries(
        state_stratified,
        ("dataset_group", "layer", "full_correct", "score_type", "state"),
        "value",
        bootstrap_replicates,
        seed,
    )
    effect_summary = grouped_summaries(
        effect_stratified,
        ("dataset_group", "layer", "full_correct", "score_type", "effect"),
        "value",
        bootstrap_replicates,
        seed,
    )
    thresholds = threshold_fraction_rows(effect_stratified)
    behavior_counts = behavior_count_rows(branch_rows)
    likelihood_behavior = likelihood_behavior_rows(branch_rows, correctness_threshold)
    sample_averages = sample_layer_averages(effect_rows)
    sample_summary = grouped_summaries(
        add_full_correct_strata(sample_averages),
        ("dataset_group", "full_correct", "score_type", "effect"),
        "value",
        bootstrap_replicates,
        seed,
    )
    answer_length_summary = grouped_summaries(
        add_full_correct_strata(sample_averages),
        ("dataset_group", "answer_length_bin", "full_correct", "score_type", "effect"),
        "value",
        bootstrap_replicates,
        seed,
    )
    answer_length_correlations = answer_length_correlation_rows(sample_averages)
    agreement = agreement_rows(effect_rows)

    write_csv(output_dir / "layer_state_distributions.csv", state_summary)
    write_csv(output_dir / "layer_signed_effects.csv", effect_summary)
    write_csv(output_dir / "layer_threshold_fractions.csv", thresholds)
    write_csv(output_dir / "greedy_behavior_counts.csv", behavior_counts)
    write_csv(output_dir / "likelihood_behavior_relation.csv", likelihood_behavior)
    write_csv(output_dir / "sample_aggregated_effects.csv", sample_summary)
    write_csv(output_dir / "answer_length_sensitivity.csv", answer_length_summary)
    write_csv(output_dir / "answer_length_correlations.csv", answer_length_correlations)
    write_csv(output_dir / "sequence_mean_agreement.csv", agreement)
    write_csv(output_dir / "pinned_full_relabeling.csv", relabel_rows)
    relabel_counts: dict[tuple[str, str, bool], int] = Counter(
        (row["dataset"], row["inherited_bucket"], bool(row["pinned_full_correct"]))
        for row in relabel_rows
    )
    write_csv(
        output_dir / "pinned_full_relabeling_summary.csv",
        [
            {
                "dataset": key[0],
                "inherited_bucket": key[1],
                "pinned_full_correct": key[2],
                "count": count,
            }
            for key, count in sorted(relabel_counts.items())
        ],
    )
    create_plots(
        output_dir,
        state_summary,
        effect_summary,
        thresholds,
        behavior_counts,
        likelihood_behavior,
    )

    result_ids = {row["id"] for row in results}
    candidate_ids = {row["id"] for row in read_jsonl(Path(config["candidate_manifest"]))}
    exclusions_path = result_path.parent / "technical_exclusions.json"
    exclusions_payload = json.loads(exclusions_path.read_text(encoding="utf-8"))
    excluded_ids = {row["id"] for row in exclusions_payload["exclusions"]}
    if result_ids & excluded_ids:
        raise ValueError("A sample cannot be both scored and technically excluded")
    if result_ids | excluded_ids != candidate_ids:
        raise ValueError(
            "Results plus predefined technical exclusions do not cover the fixed candidate set"
        )
    expected_layers = [int(layer) for layer in config["layer_grid"]]
    if any([int(layer["layer"]) for layer in row["layers"]] != expected_layers for row in results):
        raise ValueError("At least one result does not contain the exact frozen layer grid")
    missing_ids = sorted(candidate_ids - result_ids)
    full_parity_deltas = [
        abs(float(layer["states"]["FULL"]["sequence_logprob"] - result["baseline_full"]["sequence_logprob"]))
        for result in results
        for layer in result["layers"]
    ]
    full_generation_matches = [
        layer["states"]["FULL"]["generated_token_ids"] == result["baseline_full"]["generated_token_ids"]
        for result in results
        for layer in result["layers"]
    ]
    write_json(
        output_dir / "analysis_manifest.json",
        {
            "schema_version": "stage_b_reference_analysis_v1",
            "results_path": str(result_path),
            "results_sha256": file_sha256(result_path),
            "candidate_manifest_sha256": file_sha256(Path(config["candidate_manifest"])),
            "candidate_count": len(candidate_ids),
            "result_count": len(results),
            "technical_exclusion_count": len(excluded_ids),
            "technical_exclusions_path": str(exclusions_path),
            "missing_candidate_ids": missing_ids,
            "layer_grid": config["layer_grid"],
            "factorial_state_map": {
                "IGNORE": {"read": 0, "write": 0},
                "READ_ONLY": {"read": 1, "write": 0},
                "WRITE_ONLY": {"read": 0, "write": 1},
                "FULL": {"read": 1, "write": 1},
            },
            "epsilon_sequence": epsilon_sequence,
            "epsilon_mean": epsilon_mean,
            "correctness_rule": f"official score >= {correctness_threshold}",
            "bootstrap_unit": "sample",
            "bootstrap_replicates": bootstrap_replicates,
            "full_sequence_parity_max_abs": max(full_parity_deltas, default=math.nan),
            "full_generation_parity": all(full_generation_matches),
            "sample_layer_pairs_not_used_as_independent_prevalence_units": True,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
