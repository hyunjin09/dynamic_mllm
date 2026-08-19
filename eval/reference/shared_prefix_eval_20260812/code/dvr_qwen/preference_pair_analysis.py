from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable


NUM_LAYERS = 28
PAIR_TYPES = ("correctness", "efficiency")
BENCHMARKS = ("gqa", "textvqa", "chartqa", "docvqa")


def pair_geometry(chosen_mask: str, rejected_mask: str) -> dict[str, float | int | str]:
    for name, mask in (("chosen", chosen_mask), ("rejected", rejected_mask)):
        if len(mask) != NUM_LAYERS:
            raise ValueError(f"{name} mask must be a {NUM_LAYERS}-bit string")
        if set(mask) - {"0", "1"}:
            raise ValueError(f"{name} mask must be binary")
    chosen = {i for i, bit in enumerate(chosen_mask) if bit == "1"}
    rejected = {i for i, bit in enumerate(rejected_mask) if bit == "1"}
    chosen_only = chosen - rejected
    rejected_only = rejected - chosen
    union = chosen | rejected
    hamming = len(chosen_only) + len(rejected_only)
    if not chosen_only and not rejected_only:
        set_relation = "identical"
    elif not chosen_only:
        set_relation = "chosen_subset"
    elif not rejected_only:
        set_relation = "rejected_subset"
    else:
        set_relation = "cross_swap"
    return {
        "chosen_budget": len(chosen),
        "rejected_budget": len(rejected),
        "budget_gap": len(rejected) - len(chosen),
        "chosen_only_on": len(chosen_only),
        "rejected_only_on": len(rejected_only),
        "hamming": hamming,
        "normalized_hamming": hamming / NUM_LAYERS,
        "jaccard": len(chosen & rejected) / len(union) if union else 1.0,
        "set_relation": set_relation,
    }


@dataclass
class DiscreteStats:
    counts: Counter[float] = field(default_factory=Counter)
    n: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def add(self, value: float | int) -> None:
        number = float(value)
        self.counts[number] += 1
        self.n += 1
        self.total += number
        self.total_sq += number * number

    def _value_at(self, index: int) -> float:
        seen = 0
        for value in sorted(self.counts):
            seen += self.counts[value]
            if index < seen:
                return value
        raise IndexError(index)

    def quantile(self, q: float) -> float | None:
        if not self.n:
            return None
        position = (self.n - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        low_value = self._value_at(lower)
        if lower == upper:
            return low_value
        high_value = self._value_at(upper)
        return low_value + (high_value - low_value) * (position - lower)

    def summary(self) -> dict[str, float | int | None]:
        if not self.n:
            return {key: None for key in ("n", "mean", "std", "min", "q25", "median", "q75", "max")}
        variance = max(0.0, self.total_sq / self.n - (self.total / self.n) ** 2)
        return {
            "n": self.n,
            "mean": self.total / self.n,
            "std": math.sqrt(variance),
            "min": min(self.counts),
            "q25": self.quantile(0.25),
            "median": self.quantile(0.5),
            "q75": self.quantile(0.75),
            "max": max(self.counts),
        }


def effective_pair_weight_mass(
    *,
    recommended_weights: dict[str, float],
    loss_multipliers: dict[str, float],
    sampled_pairs_per_type: dict[str, int],
) -> dict[str, dict[str, float]]:
    raw = {
        pair_type: float(recommended_weights[pair_type])
        * float(loss_multipliers[pair_type])
        * int(sampled_pairs_per_type.get(pair_type, 0))
        for pair_type in PAIR_TYPES
    }
    total = sum(raw.values())
    normalized = {key: value / total if total else 0.0 for key, value in raw.items()}
    return {"raw_weight": raw, "normalized_mass": normalized}


@dataclass
class PairGroup:
    sample_ids: set[str] = field(default_factory=set)
    pair_count: int = 0
    subtypes: Counter[str] = field(default_factory=Counter)
    budget_signs: Counter[str] = field(default_factory=Counter)
    set_relations: Counter[str] = field(default_factory=Counter)
    chosen_budget: DiscreteStats = field(default_factory=DiscreteStats)
    rejected_budget: DiscreteStats = field(default_factory=DiscreteStats)
    budget_gap: DiscreteStats = field(default_factory=DiscreteStats)
    hamming: DiscreteStats = field(default_factory=DiscreteStats)
    normalized_hamming: DiscreteStats = field(default_factory=DiscreteStats)
    jaccard: DiscreteStats = field(default_factory=DiscreteStats)
    chosen_only_on: DiscreteStats = field(default_factory=DiscreteStats)
    rejected_only_on: DiscreteStats = field(default_factory=DiscreteStats)

    def add(self, row: dict[str, Any], geometry: dict[str, float | int | str]) -> None:
        self.sample_ids.add(str(row["uid"]))
        self.pair_count += 1
        self.subtypes[str(row["pair_subtype"])] += 1
        gap = int(geometry["budget_gap"])
        self.budget_signs["negative" if gap < 0 else "positive" if gap > 0 else "zero"] += 1
        self.set_relations[str(geometry["set_relation"])] += 1
        for name in (
            "chosen_budget",
            "rejected_budget",
            "budget_gap",
            "hamming",
            "normalized_hamming",
            "jaccard",
            "chosen_only_on",
            "rejected_only_on",
        ):
            getattr(self, name).add(geometry[name])


@dataclass
class SamplePairs:
    total: int = 0
    by_type: Counter[str] = field(default_factory=Counter)
    by_subtype: Counter[str] = field(default_factory=Counter)
    referenced_masks: set[str] = field(default_factory=set)
    chosen_masks: set[str] = field(default_factory=set)
    rejected_masks: set[str] = field(default_factory=set)
    budget_gap: dict[str, DiscreteStats] = field(
        default_factory=lambda: {pair_type: DiscreteStats() for pair_type in PAIR_TYPES}
    )
    hamming: dict[str, DiscreteStats] = field(
        default_factory=lambda: {pair_type: DiscreteStats() for pair_type in PAIR_TYPES}
    )
    jaccard: dict[str, DiscreteStats] = field(
        default_factory=lambda: {pair_type: DiscreteStats() for pair_type in PAIR_TYPES}
    )

    def add(self, row: dict[str, Any], geometry: dict[str, float | int | str]) -> None:
        pair_type = str(row["pair_type"])
        self.total += 1
        self.by_type[pair_type] += 1
        self.by_subtype[f"{pair_type}:{row['pair_subtype']}"] += 1
        chosen = str(row["chosen_mask_key"])
        rejected = str(row["rejected_mask_key"])
        self.referenced_masks.update((chosen, rejected))
        self.chosen_masks.add(chosen)
        self.rejected_masks.add(rejected)
        self.budget_gap[pair_type].add(geometry["budget_gap"])
        self.hamming[pair_type].add(geometry["hamming"])
        self.jaccard[pair_type].add(round(float(geometry["jaccard"]), 12))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from exc


def _flatten_stats(prefix: str, stats: DiscreteStats) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in stats.summary().items()}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _group_key_rows(row: dict[str, Any]) -> list[tuple[str, ...]]:
    pair_type = str(row["pair_type"])
    return [
        ("overall", pair_type),
        ("split", str(row["split"]), pair_type),
        ("benchmark", str(row["benchmark"]), pair_type),
        ("source_bucket", str(row["source_bucket"]), pair_type),
        (
            "cell",
            str(row["split"]),
            str(row["benchmark"]),
            str(row["source_bucket"]),
            pair_type,
        ),
        ("subtype", pair_type, str(row["pair_subtype"])),
    ]


def _layer_group_keys(row: dict[str, Any]) -> list[tuple[str, str]]:
    pair_type = str(row["pair_type"])
    return [("overall", pair_type), (str(row["benchmark"]), pair_type)]


def _validate_pair_row(
    row: dict[str, Any], geometry: dict[str, float | int | str], expected_split: str
) -> list[str]:
    issues: list[str] = []
    pair_type = str(row.get("pair_type"))
    if str(row.get("split")) != expected_split:
        issues.append("split_mismatch")
    for field_name, geometry_name in (
        ("chosen_budget", "chosen_budget"),
        ("rejected_budget", "rejected_budget"),
        ("budget_delta_rejected_minus_chosen", "budget_gap"),
        ("hamming_distance", "hamming"),
    ):
        if int(row[field_name]) != int(geometry[geometry_name]):
            issues.append(f"{field_name}_mismatch")
    if str(row["chosen_mask_key"]) == str(row["rejected_mask_key"]):
        issues.append("identical_masks")
    if pair_type == "correctness":
        if not bool(row["chosen_correct"]) or bool(row["rejected_correct"]):
            issues.append("correctness_rule_violation")
    elif pair_type == "efficiency":
        if not bool(row["chosen_correct"]) or not bool(row["rejected_correct"]):
            issues.append("efficiency_correctness_violation")
        if int(geometry["budget_gap"]) <= 0:
            issues.append("efficiency_budget_rule_violation")
    else:
        issues.append("unknown_pair_type")
    return issues


def analyze_dataset(dataset_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {str(row["uid"]): row for row in iter_jsonl(dataset_dir / "sample_targets.jsonl")}
    sample_pairs: dict[str, SamplePairs] = defaultdict(SamplePairs)
    groups: dict[tuple[str, ...], PairGroup] = defaultdict(PairGroup)
    layer_counts: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: {
            "pairs": [0] * NUM_LAYERS,
            "chosen_on": [0] * NUM_LAYERS,
            "rejected_on": [0] * NUM_LAYERS,
            "chosen_only_on": [0] * NUM_LAYERS,
            "rejected_only_on": [0] * NUM_LAYERS,
            "both_on": [0] * NUM_LAYERS,
        }
    )
    joint_hist: dict[str, Counter[tuple[int, int]]] = defaultdict(Counter)
    audit_issues: Counter[str] = Counter()
    seen_by_file: Counter[str] = Counter()

    pair_files = (
        ("train", dataset_dir / "train_preference_pairs.jsonl"),
        ("validation", dataset_dir / "validation_preference_pairs.jsonl"),
    )
    for expected_split, path in pair_files:
        for row in iter_jsonl(path):
            seen_by_file[expected_split] += 1
            uid = str(row["uid"])
            if uid not in targets:
                audit_issues["pair_uid_missing_from_targets"] += 1
            geometry = pair_geometry(str(row["chosen_mask_key"]), str(row["rejected_mask_key"]))
            audit_issues.update(_validate_pair_row(row, geometry, expected_split))
            sample_pairs[uid].add(row, geometry)
            for key in _group_key_rows(row):
                groups[key].add(row, geometry)
            pair_type = str(row["pair_type"])
            joint_hist[pair_type][(int(geometry["budget_gap"]), int(geometry["hamming"]))] += 1
            chosen_mask = str(row["chosen_mask_key"])
            rejected_mask = str(row["rejected_mask_key"])
            for key in _layer_group_keys(row):
                counts = layer_counts[key]
                for layer, (chosen_bit, rejected_bit) in enumerate(zip(chosen_mask, rejected_mask)):
                    counts["pairs"][layer] += 1
                    counts["chosen_on"][layer] += chosen_bit == "1"
                    counts["rejected_on"][layer] += rejected_bit == "1"
                    counts["chosen_only_on"][layer] += chosen_bit == "1" and rejected_bit == "0"
                    counts["rejected_only_on"][layer] += chosen_bit == "0" and rejected_bit == "1"
                    counts["both_on"][layer] += chosen_bit == "1" and rejected_bit == "1"

    target_count_mismatches = 0
    sample_rows: list[dict[str, Any]] = []
    sample_group_values: dict[tuple[str, ...], dict[str, DiscreteStats]] = defaultdict(
        lambda: defaultdict(DiscreteStats)
    )
    pair_type_availability: Counter[str] = Counter()
    for uid, target in targets.items():
        observed = sample_pairs.get(uid, SamplePairs())
        expected_total = int(target.get("preference_pair_count", 0))
        expected_correctness = int(target.get("correctness_pair_count", 0))
        expected_efficiency = int(target.get("efficiency_pair_count", 0))
        if (
            observed.total != expected_total
            or observed.by_type["correctness"] != expected_correctness
            or observed.by_type["efficiency"] != expected_efficiency
        ):
            target_count_mismatches += 1
        available = [pair_type for pair_type in PAIR_TYPES if observed.by_type[pair_type] > 0]
        availability = "+".join(available) if available else "none"
        pair_type_availability[availability] += 1
        sample_row: dict[str, Any] = {
            "uid": uid,
            "split": target["split"],
            "benchmark": target["benchmark"],
            "source_bucket": target["source_bucket"],
            "training_eligible": target["training_eligible"],
            "candidate_routes": target["candidate_route_count"],
            "correct_routes": target["correct_route_count"],
            "incorrect_routes": target["incorrect_route_count"],
            "minimum_correct_budget": target["minimum_correct_budget"],
            "cooptimal_masks": target["cooptimal_route_count"],
            "null_visual_optimal": target["null_visual_optimal"],
            "pairs_total": observed.total,
            "correctness_pairs": observed.by_type["correctness"],
            "efficiency_pairs": observed.by_type["efficiency"],
            "unique_referenced_masks": len(observed.referenced_masks),
            "unique_chosen_masks": len(observed.chosen_masks),
            "unique_rejected_masks": len(observed.rejected_masks),
            "pair_type_availability": availability,
        }
        for pair_type in PAIR_TYPES:
            for metric_name, stats in (
                ("budget_gap", observed.budget_gap[pair_type]),
                ("hamming", observed.hamming[pair_type]),
                ("jaccard", observed.jaccard[pair_type]),
            ):
                summary = stats.summary()
                sample_row[f"{pair_type}_{metric_name}_mean"] = summary["mean"]
                sample_row[f"{pair_type}_{metric_name}_median"] = summary["median"]
        sample_rows.append(sample_row)
        if observed.total:
            keys = [
                ("overall",),
                ("split", str(target["split"])),
                ("benchmark", str(target["benchmark"])),
                ("source_bucket", str(target["source_bucket"])),
                (
                    "cell",
                    str(target["split"]),
                    str(target["benchmark"]),
                    str(target["source_bucket"]),
                ),
            ]
            values = {
                "pairs_total": observed.total,
                "correctness_pairs": observed.by_type["correctness"],
                "efficiency_pairs": observed.by_type["efficiency"],
                "unique_referenced_masks": len(observed.referenced_masks),
                "candidate_routes": int(target["candidate_route_count"]),
                "correct_routes": int(target["correct_route_count"]),
                "incorrect_routes": int(target["incorrect_route_count"]),
                "cooptimal_masks": int(target["cooptimal_route_count"]),
            }
            if target["minimum_correct_budget"] is not None:
                values["minimum_correct_budget"] = int(target["minimum_correct_budget"])
            for key in keys:
                for name, value in values.items():
                    sample_group_values[key][name].add(value)

    group_rows: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        row: dict[str, Any] = {"group_kind": key[0], "group": "|".join(key[1:])}
        row.update({"samples": len(group.sample_ids), "pairs": group.pair_count})
        for name in (
            "chosen_budget",
            "rejected_budget",
            "budget_gap",
            "hamming",
            "normalized_hamming",
            "jaccard",
            "chosen_only_on",
            "rejected_only_on",
        ):
            row.update(_flatten_stats(name, getattr(group, name)))
        for category, count in sorted(group.budget_signs.items()):
            row[f"budget_gap_{category}_count"] = count
            row[f"budget_gap_{category}_rate"] = count / group.pair_count
        for category, count in sorted(group.set_relations.items()):
            row[f"set_relation_{category}_count"] = count
            row[f"set_relation_{category}_rate"] = count / group.pair_count
        group_rows.append(row)

    sample_group_rows: list[dict[str, Any]] = []
    for key, metrics in sorted(sample_group_values.items()):
        row = {"group_kind": key[0], "group": "|".join(key[1:]) or "all"}
        for metric_name, stats in metrics.items():
            row.update(_flatten_stats(metric_name, stats))
        sample_group_rows.append(row)

    layer_rows: list[dict[str, Any]] = []
    for (benchmark, pair_type), metrics in sorted(layer_counts.items()):
        for layer in range(NUM_LAYERS):
            pairs = metrics["pairs"][layer]
            differing = metrics["chosen_only_on"][layer] + metrics["rejected_only_on"][layer]
            layer_rows.append(
                {
                    "benchmark": benchmark,
                    "pair_type": pair_type,
                    "layer": layer,
                    "pairs": pairs,
                    "chosen_on_rate": metrics["chosen_on"][layer] / pairs,
                    "rejected_on_rate": metrics["rejected_on"][layer] / pairs,
                    "chosen_only_on_rate": metrics["chosen_only_on"][layer] / pairs,
                    "rejected_only_on_rate": metrics["rejected_only_on"][layer] / pairs,
                    "difference_rate": differing / pairs,
                    "direction_among_differences": (
                        (metrics["chosen_only_on"][layer] - metrics["rejected_only_on"][layer]) / differing
                        if differing
                        else 0.0
                    ),
                }
            )

    _write_csv(output_dir / "pair_group_stats.csv", group_rows)
    _write_csv(output_dir / "sample_group_stats.csv", sample_group_rows)
    _write_csv(output_dir / "per_sample_stats.csv", sample_rows)
    _write_csv(output_dir / "layer_pair_signal.csv", layer_rows)

    trainer_mass = effective_pair_weight_mass(
        recommended_weights={"correctness": 1.0, "efficiency": 0.5},
        loss_multipliers={"correctness": 1.0, "efficiency": 0.1},
        sampled_pairs_per_type={"correctness": 1, "efficiency": 1},
    )
    overall_groups = {
        pair_type: _pair_group_to_json(groups[("overall", pair_type)]) for pair_type in PAIR_TYPES
    }
    summary = {
        "dataset_dir": str(dataset_dir),
        "population": {
            "targets": len(targets),
            "eligible_samples": sum(bool(row["training_eligible"]) for row in targets.values()),
            "samples_with_pairs": len(sample_pairs),
            "train_pairs": seen_by_file["train"],
            "validation_pairs": seen_by_file["validation"],
            "total_pairs": sum(seen_by_file.values()),
            "pair_type_availability_all_targets": dict(sorted(pair_type_availability.items())),
        },
        "overall_by_pair_type": overall_groups,
        "audit": {
            "row_issue_counts": dict(sorted(audit_issues.items())),
            "target_pair_count_mismatch_samples": target_count_mismatches,
            "passed": not audit_issues and target_count_mismatches == 0,
        },
        "current_trajectory_trainer_nominal_two_pair_step": {
            "scope": "UIDs having both pair types; source-bucket multiplier 2.0 cancels after normalization",
            "recommended_weights": {"correctness": 1.0, "efficiency": 0.5},
            "loss_multipliers": {"correctness": 1.0, "efficiency": 0.1},
            **trainer_mass,
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    _make_figures(output_dir, groups, sample_rows, joint_hist, layer_rows)
    return summary


def _pair_group_to_json(group: PairGroup) -> dict[str, Any]:
    return {
        "samples": len(group.sample_ids),
        "pairs": group.pair_count,
        "subtypes": dict(sorted(group.subtypes.items())),
        "budget_gap_sign_counts": dict(sorted(group.budget_signs.items())),
        "budget_gap_sign_rates": {
            key: value / group.pair_count for key, value in sorted(group.budget_signs.items())
        },
        "set_relation_counts": dict(sorted(group.set_relations.items())),
        "set_relation_rates": {
            key: value / group.pair_count for key, value in sorted(group.set_relations.items())
        },
        **{
            name: getattr(group, name).summary()
            for name in (
                "chosen_budget",
                "rejected_budget",
                "budget_gap",
                "hamming",
                "normalized_hamming",
                "jaccard",
                "chosen_only_on",
                "rejected_only_on",
            )
        },
    }


def _make_figures(
    output_dir: Path,
    groups: dict[tuple[str, ...], PairGroup],
    sample_rows: list[dict[str, Any]],
    joint_hist: dict[str, Counter[tuple[int, int]]],
    layer_rows: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"correctness": "#2274A5", "efficiency": "#D95F02"}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    for axis, pair_type in zip(axes, PAIR_TYPES):
        stats = groups[("overall", pair_type)].budget_gap
        xs = sorted(stats.counts)
        ys = [100 * stats.counts[x] / stats.n for x in xs]
        axis.bar(xs, ys, color=colors[pair_type], width=0.85)
        axis.axvline(0, color="#333333", linewidth=0.8)
        axis.set_title(pair_type.capitalize())
        axis.set_xlabel("Budget gap (rejected - chosen)")
        axis.set_ylabel("Pairs (%)")
    fig.savefig(output_dir / "figure_1_budget_gap_distribution.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    for axis, pair_type in zip(axes, PAIR_TYPES):
        x = np.arange(len(BENCHMARKS))
        medians, low, high = [], [], []
        for benchmark in BENCHMARKS:
            summary = groups[("benchmark", benchmark, pair_type)].jaccard.summary()
            medians.append(summary["median"])
            low.append(summary["median"] - summary["q25"])
            high.append(summary["q75"] - summary["median"])
        axis.errorbar(x, medians, yerr=np.array([low, high]), fmt="o", capsize=4, color=colors[pair_type])
        axis.set_xticks(x, [name.upper() for name in BENCHMARKS], rotation=20)
        axis.set_ylim(-0.02, 1.02)
        axis.set_ylabel("Mask Jaccard (median, IQR)")
        axis.set_title(pair_type.capitalize())
    fig.savefig(output_dir / "figure_2_mask_similarity_by_benchmark.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)
    for axis, pair_type in zip(axes, PAIR_TYPES):
        histogram = joint_hist[pair_type]
        gaps = sorted({key[0] for key in histogram})
        matrix = np.zeros((NUM_LAYERS + 1, len(gaps)), dtype=float)
        for (gap, hamming), count in histogram.items():
            matrix[hamming, gaps.index(gap)] += count
        matrix = 100 * matrix / max(1, matrix.sum())
        image = axis.imshow(matrix, origin="lower", aspect="auto", cmap="magma")
        tick_positions = np.linspace(0, len(gaps) - 1, min(9, len(gaps)), dtype=int)
        axis.set_xticks(tick_positions, [str(gaps[i]) for i in tick_positions])
        axis.set_xlabel("Budget gap (rejected - chosen)")
        axis.set_ylabel("Hamming distance")
        axis.set_title(pair_type.capitalize())
        fig.colorbar(image, ax=axis, label="All pairs (%)")
    fig.savefig(output_dir / "figure_3_budget_gap_hamming_joint.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for axis, pair_type in zip(axes, PAIR_TYPES):
        matrix = np.zeros((len(BENCHMARKS), NUM_LAYERS), dtype=float)
        for row in layer_rows:
            if row["pair_type"] == pair_type and row["benchmark"] in BENCHMARKS:
                matrix[BENCHMARKS.index(row["benchmark"]), int(row["layer"])] = float(
                    row["direction_among_differences"]
                )
        image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        axis.set_yticks(range(len(BENCHMARKS)), [name.upper() for name in BENCHMARKS])
        axis.set_xticks(range(0, NUM_LAYERS, 3))
        axis.set_xlabel("Layer index")
        axis.set_title(pair_type.capitalize())
        fig.colorbar(image, ax=axis, label="Direction among differing pairs (+ chosen-on)")
    fig.savefig(output_dir / "figure_4_layer_direction_signal.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), constrained_layout=True)
    metrics = (
        ("pairs_total", "All pairs per sample"),
        ("correctness_pairs", "Correctness pairs per sample"),
        ("efficiency_pairs", "Efficiency pairs per sample"),
    )
    eligible_rows = [row for row in sample_rows if row["pairs_total"]]
    for axis, (metric, title) in zip(axes, metrics):
        values = [[row[metric] for row in eligible_rows if row["benchmark"] == benchmark] for benchmark in BENCHMARKS]
        axis.boxplot(values, tick_labels=[name.upper() for name in BENCHMARKS], showfliers=False)
        axis.tick_params(axis="x", rotation=20)
        axis.set_title(title)
        axis.set_ylabel("Count")
    fig.savefig(output_dir / "figure_5_pairs_per_sample.png", dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze correctness-first preference pair geometry")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze_dataset(args.dataset_dir, args.output_dir)
    print(json.dumps(summary["population"], indent=2, sort_keys=True))
    print(json.dumps(summary["audit"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
