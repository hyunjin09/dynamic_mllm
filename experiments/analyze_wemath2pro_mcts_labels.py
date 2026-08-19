#!/usr/bin/env python3
"""Audit WeMath2.0-Pro cap-400 MCTS labels for training suitability."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.analyze_mcts_bce_labels import (
    ALL_OFF,
    ALL_ON,
    NUM_LAYERS,
    fmean_defined,
    mask_entropy,
    mask_key,
    oracle_metrics,
    quantile_summary,
    sha256_file,
    summarize_masks,
    svg_bars,
    svg_histogram,
    svg_lines,
    write_csv,
    write_json,
    write_jsonl,
)
from label_regeneration.bce_geometry import (
    Mask,
    as_mask,
    hamming,
    layer_marginals,
    pareto_efficient_indices,
    polar_route_weights,
)
from label_regeneration.derived import select_diverse_valid_routes


PROJECT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT / "outputs/label_regeneration/wemath2pro_cap400_v2"
OUTPUT_ROOT = PROJECT / "outputs/wemath2pro_mcts_label_analysis_v1"
SELECTION_SEED = 20260809
ROUTE_CAP = 50


def classify_group(current_status: str, valid_masks: list[Mask]) -> str:
    if current_status == "wrong":
        return "A" if valid_masks else "D"
    if current_status != "correct":
        raise ValueError(f"unexpected FULL status {current_status!r}")
    return "B" if any(sum(mask) < NUM_LAYERS for mask in valid_masks) else "C"


def validate_record(
    record: dict[str, Any],
    manifest: dict[str, Any],
    *,
    accepted_contracts: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    uid = manifest["uid"]
    sample = record.get("sample", {})
    if sample.get("uid") != uid:
        raise ValueError(f"UID mismatch for {uid}")
    if sample.get("image_content_sha256") != manifest["image_content_sha256"]:
        raise ValueError(f"image checksum binding mismatch for {uid}")
    if sample.get("question") != manifest["question"] or sample.get("answer") != manifest["answer"]:
        raise ValueError(f"question/answer binding mismatch for {uid}")
    runtime = record.get("runtime", {})
    contract = runtime.get("contract_sha256")
    resume_contract = runtime.get("resume_compatible_contract_sha256")
    if contract not in accepted_contracts and resume_contract not in accepted_contracts:
        raise ValueError(f"incompatible execution contract for {uid}: {contract}")
    requested = int(record["mcts"]["requested_simulations"])
    completed = int(record["mcts"]["completed_simulations"])
    if requested not in (200, 400) or completed != requested:
        raise ValueError(f"nonterminal or above-cap search budget for {uid}: {completed}/{requested}")
    expected_budget = 200 if sample["current_all_on_status"] == "correct" else 400
    if requested != expected_budget:
        raise ValueError(f"budget does not match current FULL status for {uid}")
    candidates = record.get("candidate_executions")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError(f"missing candidate executions for {uid}")
    masks: set[Mask] = set()
    route_ids: set[str] = set()
    by_mask: dict[Mask, dict[str, Any]] = {}
    for route in candidates:
        mask = as_mask(route["visual_on_mask"])
        if len(mask) != NUM_LAYERS or route["mask_key"] != mask_key(mask):
            raise ValueError(f"malformed mask for {uid}")
        if mask in masks or route["route_id"] in route_ids:
            raise ValueError(f"duplicate route/mask for {uid}")
        masks.add(mask)
        route_ids.add(route["route_id"])
        by_mask[mask] = route
        if int(route["num_visual_on_layers"]) != sum(mask):
            raise ValueError(f"ON count mismatch for {uid}")
        expected_valid = float(route["score"]) >= float(route["correctness_threshold"])
        if bool(route["result_correct"]) != expected_valid:
            raise ValueError(f"validity/threshold mismatch for {uid}")
        if route.get("scoring_timed_out") and (float(route["score"]) != 0.0 or route["result_correct"]):
            raise ValueError(f"timeout was not conservatively invalid for {uid}")
    if ALL_ON not in by_mask or ALL_OFF not in by_mask:
        raise ValueError(f"missing ALL-ON/OFF anchor for {uid}")
    simulations = record["mcts"].get("simulations")
    if not isinstance(simulations, list) or len(simulations) != requested:
        raise ValueError(f"simulation trace mismatch for {uid}")
    simulated_masks = {as_mask(row["evaluated_mask"]) for row in simulations}
    expected_masks = {ALL_ON, ALL_OFF} | simulated_masks
    if masks != expected_masks:
        raise ValueError(f"candidate/simulation mask linkage mismatch for {uid}")
    evaluated = record["mcts"].get("evaluated_masks")
    if not isinstance(evaluated, list):
        raise ValueError(f"missing evaluated-mask index for {uid}")
    evaluated_masks = {as_mask(row["visual_on_mask"]) for row in evaluated}
    evaluated_ids = {row["route_id"] for row in evaluated}
    if evaluated_masks != masks or evaluated_ids != route_ids:
        raise ValueError(f"evaluated-mask/candidate linkage mismatch for {uid}")
    root = by_mask[ALL_ON]
    if (
        float(root["score"]) != float(sample["current_all_on_score"])
        or root["prediction"] != sample["current_all_on_prediction"]
        or bool(root["result_correct"]) != (sample["current_all_on_status"] == "correct")
    ):
        raise ValueError(f"FULL anchor mismatch for {uid}")
    valid = [route for route in candidates if route["result_correct"]]
    if set(record.get("successful_route_ids", [])) != {route["route_id"] for route in valid}:
        raise ValueError(f"successful route index mismatch for {uid}")
    return candidates, valid


def analyze_record(
    record: dict[str, Any],
    manifest: dict[str, Any],
    *,
    accepted_contracts: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates, valid_routes = validate_record(record, manifest, accepted_contracts=accepted_contracts)
    raw_masks = [as_mask(route["visual_on_mask"]) for route in valid_routes]
    selected_routes = select_diverse_valid_routes(
        valid_routes, limit=ROUTE_CAP, seed=SELECTION_SEED, uid=manifest["uid"]
    ) if valid_routes else []
    selected_masks = [as_mask(route["visual_on_mask"]) for route in selected_routes]
    group = classify_group(record["sample"]["current_all_on_status"], raw_masks)
    row: dict[str, Any] = {
        "uid": manifest["uid"],
        "difficulty": str(manifest.get("difficulty") or "unknown"),
        "image_group_id": manifest["image_group_id"],
        "group": group,
        "current_all_on_status": record["sample"]["current_all_on_status"],
        "requested_simulations": int(record["mcts"]["requested_simulations"]),
        "evaluated_routes": len(candidates),
        "raw_valid_routes": len(raw_masks),
        "selected_valid_routes": len(selected_masks),
        "route_cap_applied": len(raw_masks) > ROUTE_CAP,
        "scoring_timeout_unique_predictions": int(record["sample"].get("scoring_timeout_count", 0)),
        "scoring_timeout_route_occurrences": sum(bool(route.get("scoring_timed_out")) for route in candidates),
        "actual_text_tokens": int(record["sample"]["actual_text_tokens"]),
        "actual_visual_tokens": int(record["sample"]["actual_visual_tokens"]),
        "all_off_correct": bool(next(route for route in candidates if as_mask(route["visual_on_mask"]) == ALL_OFF)["result_correct"]),
    }
    selected_payload = {
        "uid": manifest["uid"],
        "difficulty": row["difficulty"],
        "image_group_id": manifest["image_group_id"],
        "current_all_on_status": row["current_all_on_status"],
        "raw_valid_route_count": len(raw_masks),
        "selected_valid_route_count": len(selected_masks),
        "route_cap": ROUTE_CAP,
        "selection_seed": SELECTION_SEED,
        "valid_routes": [
            {
                "mask": list(mask), "mask_key": mask_key(mask), "score": float(route["score"]),
                "num_visual_on_layers": sum(mask), "num_transitions": int(route["num_transitions"]),
                "weight_equal": 1.0 / len(selected_masks) if selected_masks else None,
            }
            for route, mask in zip(selected_routes, selected_masks)
        ],
    }
    if not raw_masks:
        return row, selected_payload

    raw_geometry = summarize_masks(raw_masks)
    selected_geometry = summarize_masks(selected_masks)
    weighted = oracle_metrics(selected_masks, weighted=True)
    unweighted = oracle_metrics(selected_masks, weighted=False)
    utilities = [float(route["score"]) for route in selected_routes]
    efficient_indices = pareto_efficient_indices(selected_masks, utilities)
    efficient_masks = [selected_masks[index] for index in efficient_indices]
    pareto_oracle = oracle_metrics(efficient_masks, weighted=True)
    all_on_index = selected_masks.index(ALL_ON) if ALL_ON in selected_masks else None
    row.update({
        "raw_min_on": raw_geometry["on_count"]["minimum"],
        "raw_median_on": raw_geometry["on_count"]["median"],
        "raw_mean_on": raw_geometry["on_count"]["mean"],
        "raw_mean_pairwise_hamming": raw_geometry["pairwise_hamming"]["mean"],
        "raw_mean_bit_entropy": raw_geometry["mean_bit_entropy"],
        "raw_effective_modes_r4": raw_geometry["effective_modes_r4"],
        "selected_mean_on": selected_geometry["on_count"]["mean"],
        "selected_mean_pairwise_hamming": selected_geometry["pairwise_hamming"]["mean"],
        "selected_mean_bit_entropy": selected_geometry["mean_bit_entropy"],
        "selected_effective_modes_r4": selected_geometry["effective_modes_r4"],
        "raw_layer_marginals": raw_geometry["layer_marginals"],
        "selected_unweighted_marginals": unweighted["marginals"],
        "selected_weighted_marginals": weighted["marginals"],
        "weighted_mean_entropy": statistics.fmean(weighted["entropy"]),
        "weighted_near_tie_fraction": sum(0.45 <= q <= 0.55 for q in weighted["marginals"]) / NUM_LAYERS,
        "weighted_oracle_key": weighted["mask_key"],
        "weighted_oracle_on": weighted["on_count"],
        "weighted_oracle_all_on": weighted["all_on"],
        "weighted_oracle_valid": weighted["valid_hit_at_1"],
        "weighted_oracle_nearest_hamming": weighted["nearest_valid_hamming"],
        "unweighted_oracle_key": unweighted["mask_key"],
        "unweighted_oracle_on": unweighted["on_count"],
        "unweighted_oracle_all_on": unweighted["all_on"],
        "unweighted_oracle_valid": unweighted["valid_hit_at_1"],
        "unweighted_oracle_nearest_hamming": unweighted["nearest_valid_hamming"],
        "oracle_weighting_hamming": hamming(weighted["mask"], unweighted["mask"]),
        "pareto_efficient_routes": len(efficient_masks),
        "pareto_dominated_fraction": 1.0 - len(efficient_masks) / len(selected_masks),
        "all_on_dominated": all_on_index is not None and all_on_index not in efficient_indices,
        "pareto_oracle_on": pareto_oracle["on_count"],
        "pareto_oracle_valid": pareto_oracle["valid_hit_at_1"],
        "pareto_oracle_nearest_hamming": pareto_oracle["nearest_valid_hamming"],
    })
    weights = polar_route_weights(selected_masks)
    for route, weight in zip(selected_payload["valid_routes"], weights):
        route["weight_polar_0_3"] = weight
    return row, selected_payload


def strata(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {"overall": rows}
    for row in rows:
        for key in (
            f"difficulty:{row['difficulty']}", f"group:{row['group']}",
            f"full:{row['current_all_on_status']}", f"budget:{row['requested_simulations']}",
        ):
            output.setdefault(key, []).append(row)
    return output


def population_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for name, members in strata(rows).items():
        counts = Counter(row["group"] for row in members)
        output.append({
            "stratum": name, "records": len(members),
            "full_correct": sum(row["current_all_on_status"] == "correct" for row in members),
            "full_wrong": sum(row["current_all_on_status"] == "wrong" for row in members),
            "positive_records": sum(row["raw_valid_routes"] > 0 for row in members),
            "zero_positive_records": sum(row["raw_valid_routes"] == 0 for row in members),
            "all_off_correct": sum(row["all_off_correct"] for row in members),
            **{f"group_{group}": counts.get(group, 0) for group in "ABCD"},
        })
    return output


def geometry_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "evaluated_routes", "raw_valid_routes", "selected_valid_routes", "raw_min_on",
        "raw_median_on", "raw_mean_on", "selected_mean_on", "raw_mean_pairwise_hamming",
        "selected_mean_pairwise_hamming", "raw_mean_bit_entropy", "selected_mean_bit_entropy",
        "raw_effective_modes_r4", "selected_effective_modes_r4", "actual_text_tokens", "actual_visual_tokens",
    )
    output = []
    for name, members in strata(rows).items():
        for field in fields:
            values = [float(row[field]) for row in members if row.get(field) is not None]
            output.append({"stratum": name, "metric": field, **quantile_summary(values)})
    return output


def oracle_summary(rows: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    positive = [row for row in rows if row["raw_valid_routes"] > 0]
    output = []
    for name, members in strata(positive).items():
        counts = Counter(row[f"{prefix}_oracle_key"] for row in members)
        output.append({
            "stratum": name, "oracle": prefix, "records": len(members),
            "mean_on_layers": statistics.fmean(row[f"{prefix}_oracle_on"] for row in members),
            "all_on_fraction": statistics.fmean(row.get(f"{prefix}_oracle_all_on", False) for row in members),
            "unique_masks": len(counts), "mask_entropy_nats": mask_entropy(counts),
            "selected_valid_hit_at_1": statistics.fmean(row[f"{prefix}_oracle_valid"] for row in members),
            "mean_nearest_valid_hamming": statistics.fmean(row.get(f"{prefix}_oracle_nearest_hamming", row.get("weighted_oracle_nearest_hamming")) for row in members),
        })
    return output


def layer_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positive = [row for row in rows if row["raw_valid_routes"] > 0]
    output = []
    for name, members in strata(positive).items():
        for layer in range(NUM_LAYERS):
            output.append({
                "stratum": name, "layer": layer, "records": len(members),
                "raw_unweighted_q": statistics.fmean(row["raw_layer_marginals"][layer] for row in members),
                "selected_unweighted_q": statistics.fmean(row["selected_unweighted_marginals"][layer] for row in members),
                "selected_weighted_q": statistics.fmean(row["selected_weighted_marginals"][layer] for row in members),
            })
    return output


def pareto_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positive = [row for row in rows if row["raw_valid_routes"] > 0]
    output = []
    for name, members in strata(positive).items():
        total_routes = sum(row["selected_valid_routes"] for row in members)
        efficient = sum(row["pareto_efficient_routes"] for row in members)
        all_on = [row for row in members if row["current_all_on_status"] == "correct"]
        output.append({
            "stratum": name, "records": len(members), "total_selected_routes": total_routes,
            "total_pareto_efficient_routes": efficient,
            "route_weighted_dominated_fraction": 1.0 - efficient / total_routes,
            "mean_sample_dominated_fraction": statistics.fmean(row["pareto_dominated_fraction"] for row in members),
            "mean_efficient_routes": statistics.fmean(row["pareto_efficient_routes"] for row in members),
            "all_on_present_records": len(all_on),
            "all_on_dominated_fraction": statistics.fmean(row["all_on_dominated"] for row in all_on) if all_on else None,
            "original_oracle_mean_on": statistics.fmean(row["weighted_oracle_on"] for row in members),
            "pareto_oracle_mean_on": statistics.fmean(row["pareto_oracle_on"] for row in members),
            "original_oracle_hit_at_1": statistics.fmean(row["weighted_oracle_valid"] for row in members),
            "pareto_oracle_hit_at_1": statistics.fmean(row["pareto_oracle_valid"] for row in members),
            "pareto_oracle_mean_nearest_hamming": statistics.fmean(row["pareto_oracle_nearest_hamming"] for row in members),
        })
    return output


def timeout_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for name, members in strata(rows).items():
        output.append({
            "stratum": name, "records": len(members),
            "records_with_timeout": sum(row["scoring_timeout_unique_predictions"] > 0 for row in members),
            "unique_prediction_timeouts": sum(row["scoring_timeout_unique_predictions"] for row in members),
            "route_occurrences_marked_timeout": sum(row["scoring_timeout_route_occurrences"] for row in members),
        })
    return output


def knowledge_point_summary(manifest_rows: list[dict[str, Any]], sample_by_uid: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    positive: Counter[str] = Counter()
    full_correct: Counter[str] = Counter()
    for item in manifest_rows:
        row = sample_by_uid[item["uid"]]
        for point in item.get("knowledge_points", []):
            name = str(point)
            counts[name] += 1
            positive[name] += int(row["raw_valid_routes"] > 0)
            full_correct[name] += int(row["current_all_on_status"] == "correct")
    return [
        {"knowledge_point": name, "records": count, "positive_records": positive[name],
         "positive_fraction": positive[name] / count, "full_correct_fraction": full_correct[name] / count}
        for name, count in counts.most_common()
    ]


def create_figures(rows: list[dict[str, Any]], layer_rows: list[dict[str, Any]], pareto_rows: list[dict[str, Any]]) -> list[Path]:
    figure_dir = OUTPUT_ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    positive = [row for row in rows if row["raw_valid_routes"] > 0]
    paths = []
    for name, values, title, bins in (
        ("01_valid_route_count.svg", [row["raw_valid_routes"] for row in rows], "WeMath raw valid routes per sample", 40),
        ("02_minimum_on_count.svg", [row["raw_min_on"] for row in positive], "WeMath minimum valid-route ON layers", 29),
        ("03_pairwise_hamming.svg", [row["raw_mean_pairwise_hamming"] for row in positive if row["raw_mean_pairwise_hamming"] is not None], "WeMath per-sample mean pairwise Hamming", 29),
        ("04_bce_oracle_on_count.svg", [row["weighted_oracle_on"] for row in positive], "WeMath weighted BCE oracle ON layers", 29),
        ("05_bce_oracle_nearest_hamming.svg", [row["weighted_oracle_nearest_hamming"] for row in positive], "WeMath BCE oracle nearest-valid Hamming", 29),
    ):
        path = figure_dir / name; svg_histogram(path, values, title, bins=bins); paths.append(path)
    overall_layers = [row for row in layer_rows if row["stratum"] == "overall"]
    path = figure_dir / "06_layer_marginals.svg"
    svg_lines(path, {
        "raw": [row["raw_unweighted_q"] for row in overall_layers],
        "selected": [row["selected_unweighted_q"] for row in overall_layers],
        "weighted": [row["selected_weighted_q"] for row in overall_layers],
    }, "WeMath per-layer ON marginals"); paths.append(path)
    overall = next(row for row in pareto_rows if row["stratum"] == "overall")
    path = figure_dir / "07_pareto_comparison.svg"
    svg_bars(path, ["mean ON", "Hit@1"], {
        "original": [overall["original_oracle_mean_on"], overall["original_oracle_hit_at_1"]],
        "Pareto": [overall["pareto_oracle_mean_on"], overall["pareto_oracle_hit_at_1"]],
    }, "WeMath original versus Pareto oracle", y_max=28); paths.append(path)
    return paths


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = SOURCE_ROOT / "manifest/wemath2pro_valid_mcts_v1.jsonl"
    contract_path = SOURCE_ROOT / "frozen_execution_contract_cap400_v5.json"
    resume_audit_path = SOURCE_ROOT / "cap400_resume_audit_v1.json"
    manifest_rows = [json.loads(line) for line in manifest_path.open() if line.strip()]
    if len(manifest_rows) != 4544 or len({row["uid"] for row in manifest_rows}) != 4544:
        raise ValueError("frozen WeMath manifest is not exactly 4,544 unique UIDs")
    manifest_by_uid = {row["uid"]: row for row in manifest_rows}
    contract = json.load(contract_path.open())
    accepted_contracts = {contract["contract_sha256"], *contract["supersedes_compatible_completed_record_contracts"]}
    sample_paths = sorted((SOURCE_ROOT / "raw_route_cache").glob("shard_*/samples/*.json"))
    if len(sample_paths) != 4544:
        raise ValueError(f"expected 4,544 sample records, found {len(sample_paths)}")

    sample_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    cache_index: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, path in enumerate(sample_paths, start=1):
        payload = path.read_bytes()
        record_sha = hashlib.sha256(payload).hexdigest()
        record = json.loads(payload)
        uid = record.get("sample", {}).get("uid")
        if uid not in manifest_by_uid or uid in seen:
            raise ValueError(f"unknown or duplicate UID {uid!r}")
        seen.add(uid)
        row, selected = analyze_record(record, manifest_by_uid[uid], accepted_contracts=accepted_contracts)
        sample_rows.append(row)
        selected_rows.append(selected)
        cache_index.append({"uid": uid, "record_path": str(path.relative_to(PROJECT)), "record_sha256": record_sha,
                            "requested_simulations": row["requested_simulations"]})
        if index % 250 == 0:
            print(f"audited/analyzed {index}/4544", flush=True)
    if seen != set(manifest_by_uid):
        raise ValueError(f"missing UIDs after cache traversal: {len(set(manifest_by_uid) - seen)}")

    population = population_summary(sample_rows)
    geometry = geometry_summary(sample_rows)
    weighted = oracle_summary(sample_rows, "weighted")
    unweighted = oracle_summary(sample_rows, "unweighted")
    layers = layer_summary(sample_rows)
    pareto = pareto_summary(sample_rows)
    timeouts = timeout_summary(sample_rows)
    by_uid = {row["uid"]: row for row in sample_rows}
    knowledge = knowledge_point_summary(manifest_rows, by_uid)
    write_jsonl(OUTPUT_ROOT / "cache_record_index_v1.jsonl", cache_index)
    write_jsonl(OUTPUT_ROOT / "per_sample_training_suitability_v1.jsonl", sample_rows)
    write_jsonl(OUTPUT_ROOT / "diagnostic_selected_max50_v1.jsonl", selected_rows)
    write_csv(OUTPUT_ROOT / "population_summary.csv", population)
    write_csv(OUTPUT_ROOT / "route_geometry_summary.csv", geometry)
    write_csv(OUTPUT_ROOT / "weighted_bce_oracle_summary.csv", weighted)
    write_csv(OUTPUT_ROOT / "unweighted_bce_oracle_summary.csv", unweighted)
    write_csv(OUTPUT_ROOT / "layer_marginals.csv", layers)
    write_csv(OUTPUT_ROOT / "pareto_summary.csv", pareto)
    write_csv(OUTPUT_ROOT / "scoring_timeout_summary.csv", timeouts)
    write_csv(OUTPUT_ROOT / "knowledge_point_summary.csv", knowledge)
    figures = create_figures(sample_rows, layers, pareto)

    image_counts = Counter(row["image_group_id"] for row in sample_rows)
    audit = {
        "schema_version": "wemath2pro_cap400_completion_audit_v1",
        "status": "PASS",
        "manifest_records": 4544, "sample_records": 4544, "unique_uids": len(seen),
        "unique_image_groups": len(image_counts),
        "repeated_image_groups": sum(count > 1 for count in image_counts.values()),
        "maximum_questions_per_image": max(image_counts.values()),
        "accepted_contracts": sorted(accepted_contracts),
        "active_contract": contract["contract_sha256"],
        "budgets": dict(sorted(Counter(row["requested_simulations"] for row in sample_rows).items())),
        "candidate_routes": sum(row["evaluated_routes"] for row in sample_rows),
        "technical_invalid_source_records": 8,
        "temporary_or_error_records": 0,
        "manifest_sha256": sha256_file(manifest_path),
        "contract_sha256": sha256_file(contract_path),
        "resume_audit_sha256": sha256_file(resume_audit_path),
    }
    write_json(OUTPUT_ROOT / "completion_audit_v1.json", audit)
    source_paths = {
        "manifest": manifest_path, "contract": contract_path, "resume_audit": resume_audit_path,
        "analysis_code": Path(__file__), "geometry_code": PROJECT / "label_regeneration/bce_geometry.py",
        "selection_code": PROJECT / "label_regeneration/derived.py",
    }
    output_hashes = {
        str(path.relative_to(PROJECT)): sha256_file(path)
        for path in sorted(OUTPUT_ROOT.rglob("*")) if path.is_file()
    }
    analysis_manifest = {
        "schema_version": "wemath2pro_mcts_label_analysis_v1",
        "integrity": "PASS", "records": 4544, "route_cap": ROUTE_CAP,
        "selection_seed": SELECTION_SEED,
        "route_semantics": {"bits": 28, "one": "VISUAL_ON", "zero": "TEXT_ONLY"},
        "validity": "MathRuler score >= 1.0; timeout score 0 and invalid",
        "weighting": "POLAR ALL-ON raw weight 0.3 iff a cheaper selected valid route coexists; normalized per sample",
        "decode_rule": "q >= 0.5 resolves ON",
        "source_hashes": {name: sha256_file(path) for name, path in source_paths.items()},
        "output_hashes_before_manifest": output_hashes,
        "figures": [str(path.relative_to(PROJECT)) for path in figures],
    }
    write_json(OUTPUT_ROOT / "analysis_manifest.json", analysis_manifest)
    checksum = sha256_file(OUTPUT_ROOT / "analysis_manifest.json")
    (OUTPUT_ROOT / "analysis_manifest.json.sha256").write_text(f"{checksum}  analysis_manifest.json\n")
    print(json.dumps({"status": "PASS", "records": 4544, "output": str(OUTPUT_ROOT)}), flush=True)


if __name__ == "__main__":
    main()
