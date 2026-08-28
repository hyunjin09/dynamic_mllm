#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


NECESSARY_TAXONOMIES = {
    "read_mediated",
    "write_mediated",
    "either_removal_sufficient",
    "both_required",
}
EFFECTS = ("read_w0", "write_r0", "read_w1", "write_r1", "interaction")


def route_size_stratum(off_count: int) -> str:
    value = int(off_count)
    if value <= 4:
        return "2-4"
    if value <= 8:
        return "5-8"
    if value <= 12:
        return "9-12"
    if value <= 16:
        return "13-16"
    return ">16"


def sample_structure_category(counts: Mapping[str, int]) -> str:
    read = int(counts.get("read_mediated", 0))
    write = int(counts.get("write_mediated", 0))
    either = int(counts.get("either_removal_sufficient", 0))
    both = int(counts.get("both_required", 0))
    essential = read + write + either + both
    if essential == 0:
        return "no_essential_off"
    if both and not (read or write or either):
        return "joint_both_suppression"
    if either and not (read or write or both):
        return "either_only_ambiguous"
    if essential == 1:
        return "one_dominant_operation"
    if read and not (write or either or both):
        return "multiple_read_mediated"
    if write and not (read or either or both):
        return "multiple_write_mediated"
    return "mixed_read_write"


def cluster_bootstrap_mean(
    frame: pd.DataFrame,
    value_column: str,
    *,
    cluster_column: str = "image_group_id",
    replicates: int = 2000,
    seed: int = 20260824,
) -> dict[str, Any]:
    selected = frame[[cluster_column, value_column]].dropna()
    if selected.empty:
        raise ValueError("cannot bootstrap an empty frame")
    grouped = selected.groupby(cluster_column)[value_column].agg(["sum", "count"])
    group_sums = grouped["sum"].to_numpy(dtype=float)
    group_counts = grouped["count"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    for index in range(replicates):
        chosen = rng.integers(0, len(group_sums), size=len(group_sums))
        estimates[index] = group_sums[chosen].sum() / group_counts[chosen].sum()
    return {
        "estimate": float(selected[value_column].mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "replicates": int(replicates),
        "cluster_count": len(group_sums),
        "row_count": len(selected),
    }


def context_comparison_table(
    route_cells: pd.DataFrame,
    full_action_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = ["uid", "layer", "read_w1", "write_r1", "rescue_category", "M00", "M11"]
    full = full_action_rows[columns].drop_duplicates(["uid", "layer"]).copy()
    full = full.rename(
        columns={
            "layer": "target_layer",
            "read_w1": "full_context_read_w1",
            "write_r1": "full_context_write_r1",
            "rescue_category": "full_context_rescue_category",
            "M00": "full_context_M00",
            "M11": "full_context_M11",
        }
    )
    joined = route_cells.merge(full, on=["uid", "target_layer"], how="left", validate="one_to_one")
    if joined["full_context_rescue_category"].isna().any():
        raise ValueError("FULL-context results are missing route-conditioned sample/layer cells")
    joined["route_necessary"] = joined.taxonomy.isin(NECESSARY_TAXONOMIES)
    joined["full_context_local_rescue"] = joined.full_context_rescue_category != "no_local_rescue"
    joined["route_necessary_full_context_missed"] = (
        joined.route_necessary & ~joined.full_context_local_rescue
    )
    joined["full_context_rescue_route_redundant"] = (
        joined.full_context_local_rescue & (joined.taxonomy == "redundant")
    )
    joined["discrete_context_agreement"] = (
        joined.route_necessary == joined.full_context_local_rescue
    )
    joined["full_context_ignore_gain"] = joined.full_context_M00 - joined.full_context_M11
    joined["read_harm_sign_agreement"] = (
        (joined.read_w1 < 0) == (joined.full_context_read_w1 < 0)
    )
    joined["write_harm_sign_agreement"] = (
        (joined.write_r1 < 0) == (joined.full_context_write_r1 < 0)
    )
    necessary = joined.route_necessary
    overlap = necessary & joined.full_context_local_rescue
    full_rescue_count = int(joined.full_context_local_rescue.sum())
    necessary_count = int(necessary.sum())
    summary = {
        "matched_cell_count": len(joined),
        "route_necessary_count": necessary_count,
        "route_redundant_count": int((joined.taxonomy == "redundant").sum()),
        "full_context_local_rescue_count": full_rescue_count,
        "route_necessary_full_context_rescue_count": int(overlap.sum()),
        "route_necessary_full_context_missed_count": int(
            joined.route_necessary_full_context_missed.sum()
        ),
        "full_context_rescue_route_redundant_count": int(
            joined.full_context_rescue_route_redundant.sum()
        ),
        "discrete_context_agreement_count": int(joined.discrete_context_agreement.sum()),
        "discrete_context_agreement_fraction": float(joined.discrete_context_agreement.mean()),
        "route_necessity_recall_from_full_context": (
            float(overlap.sum() / necessary_count) if necessary_count else None
        ),
        "full_context_rescue_precision_for_route_necessity": (
            float(overlap.sum() / full_rescue_count) if full_rescue_count else None
        ),
        "route_necessary_full_context_missed_fraction": (
            float(joined.route_necessary_full_context_missed.sum() / necessary_count)
            if necessary_count
            else None
        ),
        "read_harm_sign_agreement_fraction": float(joined.read_harm_sign_agreement.mean()),
        "write_harm_sign_agreement_fraction": float(joined.write_harm_sign_agreement.mean()),
        "read_effect_pearson": float(joined.read_w1.corr(joined.full_context_read_w1)),
        "write_effect_pearson": float(joined.write_r1.corr(joined.full_context_write_r1)),
        "read_effect_spearman": float(
            joined.read_w1.rank(method="average").corr(
                joined.full_context_read_w1.rank(method="average")
            )
        ),
        "write_effect_spearman": float(
            joined.write_r1.rank(method="average").corr(
                joined.full_context_write_r1.rank(method="average")
            )
        ),
    }
    return joined, summary


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sidecar(path: Path) -> None:
    path.with_name(path.name + ".sha256").write_text(
        f"{_sha256_file(path)}  {path.name}\n", encoding="utf-8"
    )


def _write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _sidecar(path)


def _write_frame(root: Path, name: str, frame: pd.DataFrame) -> None:
    jsonl = root / f"{name}.jsonl"
    parquet = root / f"{name}.parquet"
    if jsonl.exists() or parquet.exists():
        raise FileExistsError(f"refusing to overwrite aggregate table {name}")
    frame.to_json(jsonl, orient="records", lines=True, force_ascii=False)
    frame.to_parquet(parquet, index=False)
    _sidecar(jsonl)
    _sidecar(parquet)


def _dataset_views(frame: pd.DataFrame):
    for dataset in ("gqa", "textvqa"):
        yield dataset, frame[frame.dataset == dataset]
    yield "joint", frame


def _bootstrap_indicator(frame, predicate, *, replicates=2000, seed=20260824):
    current = frame[["image_group_id"]].copy()
    current["value"] = predicate.astype(float).to_numpy()
    return cluster_bootstrap_mean(current, "value", replicates=replicates, seed=seed)


def _taxonomy_summary(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    categories = (
        "redundant",
        "read_mediated",
        "write_mediated",
        "either_removal_sufficient",
        "both_required",
    )
    for dataset, group in _dataset_views(cells):
        if group.empty:
            continue
        necessary_group = group[group.taxonomy.isin(NECESSARY_TAXONOMIES)]
        for category in categories:
            stats = _bootstrap_indicator(group, group.taxonomy == category)
            conditional = {
                "conditional_necessary_estimate": None,
                "conditional_necessary_ci_low": None,
                "conditional_necessary_ci_high": None,
                "conditional_necessary_denominator": len(necessary_group),
            }
            if category in NECESSARY_TAXONOMIES and not necessary_group.empty:
                conditional_stats = _bootstrap_indicator(
                    necessary_group,
                    necessary_group.taxonomy == category,
                )
                conditional.update(
                    {
                        "conditional_necessary_estimate": conditional_stats["estimate"],
                        "conditional_necessary_ci_low": conditional_stats["ci_low"],
                        "conditional_necessary_ci_high": conditional_stats["ci_high"],
                    }
                )
            rows.append(
                {
                    "dataset": dataset,
                    "metric": category,
                    "count": int((group.taxonomy == category).sum()),
                    **stats,
                    **conditional,
                }
            )
        stats = _bootstrap_indicator(group, group.taxonomy.isin(NECESSARY_TAXONOMIES))
        rows.append(
            {
                "dataset": dataset,
                "metric": "individually_necessary",
                "count": int(group.taxonomy.isin(NECESSARY_TAXONOMIES).sum()),
                **stats,
                "conditional_necessary_estimate": None,
                "conditional_necessary_ci_low": None,
                "conditional_necessary_ci_high": None,
                "conditional_necessary_denominator": len(necessary_group),
            }
        )
    return pd.DataFrame(rows)


def _sample_summary(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for uid, group in cells.groupby("uid", sort=True):
        counts = Counter(group.taxonomy)
        essential = sum(counts[name] for name in NECESSARY_TAXONOMIES)
        rows.append(
            {
                "uid": uid,
                "dataset": group.dataset.iloc[0],
                "image_group_id": group.image_group_id.iloc[0],
                "anchor_off_count": int(group.anchor_off_count.iloc[0]),
                "route_size_stratum": route_size_stratum(group.anchor_off_count.iloc[0]),
                "essential_off_count": int(essential),
                "redundant_off_count": int(counts["redundant"]),
                "read_mediated_count": int(counts["read_mediated"]),
                "write_mediated_count": int(counts["write_mediated"]),
                "either_removal_sufficient_count": int(counts["either_removal_sufficient"]),
                "both_required_count": int(counts["both_required"]),
                "essential_fraction": essential / len(group),
                "structure_category": sample_structure_category(counts),
            }
        )
    return pd.DataFrame(rows)


def _route_size_summary(cells: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order = ("2-4", "5-8", "9-12", "13-16", ">16")
    cells = cells.copy()
    cells["route_size_stratum"] = cells.anchor_off_count.map(route_size_stratum)
    for dataset, view in _dataset_views(cells):
        sample_view = samples if dataset == "joint" else samples[samples.dataset == dataset]
        for stratum in order:
            group = view[view.route_size_stratum == stratum]
            sample_group = sample_view[sample_view.route_size_stratum == stratum]
            if group.empty:
                continue
            necessary = group.taxonomy.isin(NECESSARY_TAXONOMIES)
            necessary_stats = _bootstrap_indicator(group, necessary, replicates=1000)
            redundant_stats = _bootstrap_indicator(group, group.taxonomy == "redundant", replicates=1000)
            rows.append(
                {
                    "dataset": dataset,
                    "route_size_stratum": stratum,
                    "sample_count": len(sample_group),
                    "off_position_count": len(group),
                    "mean_anchor_off_count": float(sample_group.anchor_off_count.mean()),
                    "mean_essential_off_count": float(sample_group.essential_off_count.mean()),
                    "necessary_fraction": necessary_stats["estimate"],
                    "necessary_ci_low": necessary_stats["ci_low"],
                    "necessary_ci_high": necessary_stats["ci_high"],
                    "redundant_fraction": redundant_stats["estimate"],
                    "redundant_ci_low": redundant_stats["ci_low"],
                    "redundant_ci_high": redundant_stats["ci_high"],
                    **{
                        f"{name}_fraction": float((group.taxonomy == name).mean())
                        for name in (
                            "read_mediated",
                            "write_mediated",
                            "either_removal_sufficient",
                            "both_required",
                        )
                    },
                }
            )
    return pd.DataFrame(rows)


def _depth_summaries(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    taxonomy_rows = []
    effect_rows = []
    categories = (
        "redundant",
        "read_mediated",
        "write_mediated",
        "either_removal_sufficient",
        "both_required",
    )
    for dataset, view in _dataset_views(cells):
        for layer in range(28):
            group = view[view.target_layer == layer]
            if group.empty:
                continue
            necessary = _bootstrap_indicator(
                group,
                group.taxonomy.isin(NECESSARY_TAXONOMIES),
                replicates=1000,
                seed=20260824 + layer,
            )
            taxonomy_rows.append(
                {
                    "dataset": dataset,
                    "layer": layer,
                    "off_position_count": len(group),
                    "necessary_fraction": necessary["estimate"],
                    "necessary_ci_low": necessary["ci_low"],
                    "necessary_ci_high": necessary["ci_high"],
                    **{
                        f"{name}_count": int((group.taxonomy == name).sum())
                        for name in categories
                    },
                    **{
                        f"{name}_fraction": float((group.taxonomy == name).mean())
                        for name in categories
                    },
                }
            )
            for effect in EFFECTS:
                stats = cluster_bootstrap_mean(
                    group,
                    effect,
                    replicates=1000,
                    seed=20260824 + 100 * layer + EFFECTS.index(effect),
                )
                effect_rows.append(
                    {
                        "dataset": dataset,
                        "layer": layer,
                        "effect": effect,
                        "median": float(group[effect].median()),
                        "negative_fraction": float((group[effect] < 0).mean()),
                        **stats,
                    }
                )
    return pd.DataFrame(taxonomy_rows), pd.DataFrame(effect_rows)


def _category_depth_comparison(
    cells: pd.DataFrame,
    *,
    replicates: int = 2000,
    seed: int = 20260824,
) -> dict[str, Any]:
    """Compare READ- and WRITE-mediated depth under image-group resampling."""
    selected = cells[
        cells.taxonomy.isin(("read_mediated", "write_mediated"))
    ][["image_group_id", "taxonomy", "target_layer"]].copy()
    read = selected[selected.taxonomy == "read_mediated"]
    write = selected[selected.taxonomy == "write_mediated"]
    output = {
        "read_mediated_count": len(read),
        "write_mediated_count": len(write),
        "read_mediated_mean_layer": float(read.target_layer.mean()) if len(read) else None,
        "write_mediated_mean_layer": float(write.target_layer.mean()) if len(write) else None,
        "read_minus_write_mean_layer": (
            float(read.target_layer.mean() - write.target_layer.mean())
            if len(read) and len(write)
            else None
        ),
        "read_minus_write_ci_low": None,
        "read_minus_write_ci_high": None,
        "replicates": int(replicates),
        "cluster_count": int(selected.image_group_id.nunique()),
    }
    if read.empty or write.empty:
        return output
    grouped = []
    for _, group in selected.groupby("image_group_id"):
        read_values = group.loc[group.taxonomy == "read_mediated", "target_layer"]
        write_values = group.loc[group.taxonomy == "write_mediated", "target_layer"]
        grouped.append(
            (
                float(read_values.sum()),
                int(read_values.count()),
                float(write_values.sum()),
                int(write_values.count()),
            )
        )
    values = np.asarray(grouped, dtype=float)
    rng = np.random.default_rng(seed)
    differences = np.full(replicates, np.nan, dtype=float)
    for index in range(replicates):
        chosen = rng.integers(0, len(values), size=len(values))
        totals = values[chosen].sum(axis=0)
        if totals[1] and totals[3]:
            differences[index] = totals[0] / totals[1] - totals[2] / totals[3]
    valid = differences[np.isfinite(differences)]
    if not len(valid):
        return output
    output["read_minus_write_ci_low"] = float(np.quantile(valid, 0.025))
    output["read_minus_write_ci_high"] = float(np.quantile(valid, 0.975))
    output["valid_bootstrap_replicates"] = int(len(valid))
    return output


def _continuous_summary(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, view in _dataset_views(cells):
        for taxonomy in ("all", *sorted(view.taxonomy.unique())):
            group = view if taxonomy == "all" else view[view.taxonomy == taxonomy]
            for effect in EFFECTS:
                stats = cluster_bootstrap_mean(group, effect, replicates=2000)
                rows.append(
                    {
                        "dataset": dataset,
                        "taxonomy": taxonomy,
                        "effect": effect,
                        "median": float(group[effect].median()),
                        "std": float(group[effect].std(ddof=0)),
                        "negative_fraction": float((group[effect] < 0).mean()),
                        **stats,
                    }
                )
    return pd.DataFrame(rows)


def _context_summaries(cells: pd.DataFrame, full: pd.DataFrame):
    joined, combined = context_comparison_table(cells, full)
    summaries = {"joint": combined}
    for dataset in ("gqa", "textvqa"):
        _, summaries[dataset] = context_comparison_table(
            cells[cells.dataset == dataset], full[full.dataset == dataset]
        )
    sample_correlations = []
    for uid, group in joined.groupby("uid"):
        if len(group) < 2:
            continue
        sample_correlations.append(
            {
                "uid": uid,
                "dataset": group.dataset.iloc[0],
                "image_group_id": group.image_group_id.iloc[0],
                "off_position_count": len(group),
                "read_spearman": group.read_w1.rank().corr(group.full_context_read_w1.rank()),
                "write_spearman": group.write_r1.rank().corr(group.full_context_write_r1.rank()),
            }
        )
    correlations = pd.DataFrame(sample_correlations)
    summaries["within_sample"] = {
        "eligible_sample_count": len(correlations),
        "mean_read_spearman": float(correlations.read_spearman.mean()),
        "median_read_spearman": float(correlations.read_spearman.median()),
        "mean_write_spearman": float(correlations.write_spearman.mean()),
        "median_write_spearman": float(correlations.write_spearman.median()),
    }
    return joined, correlations, summaries


def _save_figures(root, taxonomy, depth_taxonomy, depth_effect, route_size, samples, context):
    root.mkdir(parents=True, exist_ok=True)
    joint = taxonomy[(taxonomy.dataset == "joint") & (taxonomy.metric != "individually_necessary")]
    plt.figure(figsize=(9, 4.8))
    plt.bar(joint.metric, joint.estimate)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Fraction of anchor OFF positions")
    plt.tight_layout()
    plt.savefig(root / "necessity_taxonomy.png", dpi=180)
    plt.close()

    joint_depth = depth_taxonomy[depth_taxonomy.dataset == "joint"]
    plt.figure(figsize=(10, 5))
    for name in ("read_mediated", "write_mediated", "both_required", "redundant"):
        plt.plot(joint_depth.layer, joint_depth[f"{name}_fraction"], label=name)
    plt.xlabel("Decoder layer")
    plt.ylabel("Fraction among anchor OFF positions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "taxonomy_by_layer.png", dpi=180)
    plt.close()

    joint_effect = depth_effect[depth_effect.dataset == "joint"]
    plt.figure(figsize=(10, 5))
    for effect in EFFECTS:
        group = joint_effect[joint_effect.effect == effect]
        plt.plot(group.layer, group.estimate, label=effect)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Decoder layer")
    plt.ylabel("Mean route-conditioned effect")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(root / "continuous_effects_by_layer.png", dpi=180)
    plt.close()

    joint_size = route_size[route_size.dataset == "joint"]
    plt.figure(figsize=(8, 4.8))
    plt.bar(joint_size.route_size_stratum, joint_size.redundant_fraction)
    plt.xlabel("Anchor OFF count stratum")
    plt.ylabel("Redundant OFF-position fraction")
    plt.tight_layout()
    plt.savefig(root / "redundancy_by_route_size.png", dpi=180)
    plt.close()

    structure = samples.structure_category.value_counts(normalize=True)
    plt.figure(figsize=(9, 4.8))
    plt.bar(structure.index, structure.values)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Fraction of analyzable samples")
    plt.tight_layout()
    plt.savefig(root / "sample_mechanism_structure.png", dpi=180)
    plt.close()

    plot = context.sample(min(len(context), 5000), random_state=20260824)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].scatter(plot.full_context_read_w1, plot.read_w1, s=6, alpha=0.25)
    axes[0].set(xlabel="FULL-context READ effect", ylabel="Route-context READ effect")
    axes[1].scatter(plot.full_context_write_r1, plot.write_r1, s=6, alpha=0.25)
    axes[1].set(xlabel="FULL-context WRITE effect", ylabel="Route-context WRITE effect")
    figure.tight_layout()
    figure.savefig(root / "full_vs_route_effects.png", dpi=180)
    plt.close(figure)
    for path in root.glob("*.png"):
        _sidecar(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze full route-conditioned decomposition.")
    parser.add_argument("--config", type=Path, default=Path("configs/four_action_route_conditioned.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(config["output_root"])
    action_rows = pd.read_parquet(root / "route_conditioned_cells.parquet")
    cells = action_rows[action_rows.action == "IGNORE"].copy()
    if cells.duplicated(["uid", "target_layer"]).any():
        raise RuntimeError("route-conditioned cell table contains duplicate sample/layer rows")
    if set(cells.taxonomy) - ({"redundant"} | NECESSARY_TAXONOMIES):
        raise RuntimeError("route-conditioned cells contain unresolved/inconsistent taxonomy")
    full = pd.read_parquet(config["source_full_context_cells"])
    full = full[full.cohort == "primary_a_plus"].copy()
    taxonomy = _taxonomy_summary(cells)
    samples = _sample_summary(cells)
    route_size = _route_size_summary(cells, samples)
    depth_taxonomy, depth_effect = _depth_summaries(cells)
    depth_comparison = []
    for dataset, view in _dataset_views(cells):
        if view.empty:
            continue
        depth_comparison.append({"dataset": dataset, **_category_depth_comparison(view)})
    continuous = _continuous_summary(cells)
    context, correlations, context_summary = _context_summaries(cells, full)
    aggregates = root / "aggregate"
    aggregates.mkdir(parents=True, exist_ok=True)
    _write_frame(aggregates, "necessity_taxonomy", taxonomy)
    _write_frame(aggregates, "sample_structure", samples)
    _write_frame(aggregates, "route_size_stratification", route_size)
    _write_frame(aggregates, "depth_taxonomy", depth_taxonomy)
    _write_frame(aggregates, "depth_effects", depth_effect)
    _write_frame(aggregates, "category_depth_comparison", pd.DataFrame(depth_comparison))
    _write_frame(aggregates, "continuous_effects", continuous)
    _write_frame(aggregates, "full_context_comparison", context)
    _write_frame(aggregates, "within_sample_context_correlations", correlations)
    joint_taxonomy = taxonomy[taxonomy.dataset == "joint"].set_index("metric")
    structure_counts = samples.structure_category.value_counts().to_dict()
    summary = {
        "schema_version": "route_conditioned_aggregate_summary_v1",
        "passed": True,
        "validated_anchor_sample_count": int(cells.uid.nunique()),
        "dataset_sample_counts": {
            dataset: int(group.uid.nunique()) for dataset, group in cells.groupby("dataset")
        },
        "anchor_off_position_count": len(cells),
        "flat_action_row_count": len(action_rows),
        "taxonomy": {
            metric: {
                "count": int(row["count"]),
                "fraction": float(row["estimate"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "conditional_among_necessary_fraction": (
                    None
                    if pd.isna(row["conditional_necessary_estimate"])
                    else float(row["conditional_necessary_estimate"])
                ),
                "conditional_among_necessary_ci_low": (
                    None
                    if pd.isna(row["conditional_necessary_ci_low"])
                    else float(row["conditional_necessary_ci_low"])
                ),
                "conditional_among_necessary_ci_high": (
                    None
                    if pd.isna(row["conditional_necessary_ci_high"])
                    else float(row["conditional_necessary_ci_high"])
                ),
            }
            for metric, row in joint_taxonomy.iterrows()
        },
        "sample_structure_counts": {str(key): int(value) for key, value in structure_counts.items()},
        "context_comparison": context_summary,
        "category_depth_comparison": {
            row["dataset"]: {key: value for key, value in row.items() if key != "dataset"}
            for row in depth_comparison
        },
        "bootstrap": {
            "overall_replicates": 2000,
            "layer_and_route_size_replicates": 1000,
            "cluster": "image_group_id",
            "seed": 20260824,
        },
        "input_hashes": {
            "route_conditioned_cells": _sha256_file(root / "route_conditioned_cells.parquet"),
            "full_context_cells": _sha256_file(Path(config["source_full_context_cells"])),
            "anchor_manifest": _sha256_file(Path(config["anchor_manifest"])),
        },
        "joint_refinement_executed": False,
        "joint_refinement_decision_pending_interpretation": True,
    }
    _write_json(root / "aggregate_summary.json", summary)
    _save_figures(
        root / "figures",
        taxonomy,
        depth_taxonomy,
        depth_effect,
        route_size,
        samples,
        context,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
