from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PRIMARY = "primary_a_plus"
CONTROL_NO_CORRECTION = "control_no_correction_found"
CONTROL_VISION_REQUIRED = "control_full_correct_all_off_wrong"
EXCLUDED_ALL_OFF_RESCUE = "excluded_all_off_rescue"
EXCLUDED_OTHER = "excluded_other"


def classify_summary(row: dict[str, Any]) -> str:
    full_correct = row["current_all_on_status"] == "correct"
    all_off_correct = bool(row["all_off_correct"])
    correction_found = bool(row.get("correction_found"))
    if not full_correct and not all_off_correct and correction_found:
        return PRIMARY
    if not full_correct and not all_off_correct and not correction_found:
        return CONTROL_NO_CORRECTION
    if full_correct and not all_off_correct:
        return CONTROL_VISION_REQUIRED
    if not full_correct and all_off_correct:
        return EXCLUDED_ALL_OFF_RESCUE
    return EXCLUDED_OTHER


def evenly_spaced_ids(rows: list[dict[str, Any]], count: int) -> list[str]:
    """Choose deterministic visual-token-distributed IDs without randomness."""
    if count < 0 or count > len(rows):
        raise ValueError("selection count must be between zero and the population size")
    if count == 0:
        return []
    ordered = sorted(rows, key=lambda row: (int(row["visual_token_count"]), row["uid"]))
    if count == 1:
        return [ordered[len(ordered) // 2]["uid"]]
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise RuntimeError("evenly spaced selection produced duplicate positions")
    return [ordered[index]["uid"] for index in indices]


def shard_for(uid: str, shard_count: int = 8) -> int:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    return int.from_bytes(hashlib.sha256(uid.encode("utf-8")).digest()[:8], "big") % shard_count


def route_metadata(record: dict[str, Any]) -> dict[str, Any]:
    candidates = list(record.get("candidate_executions") or [])
    full = [1] * 28
    off = [0] * 28
    full_rows = [row for row in candidates if row.get("visual_on_mask") == full]
    off_rows = [row for row in candidates if row.get("visual_on_mask") == off]
    if len(full_rows) != 1 or len(off_rows) != 1:
        raise ValueError("raw record must contain exactly one FULL and one ALL-OFF anchor")
    correcting = [
        row
        for row in candidates
        if bool(row.get("result_correct")) and int(row.get("num_visual_on_layers", 0)) > 0
    ]
    compact = [
        {
            "route_id": row["route_id"],
            "mask": row["visual_on_mask"],
            "score": float(row["score"]),
            "hamming_distance_to_full": int(row["hamming_distance_to_all_on"]),
            "visual_on_count": int(row["num_visual_on_layers"]),
            "transition_count": int(row["num_transitions"]),
        }
        for row in correcting
    ]
    distances = [row["hamming_distance_to_full"] for row in compact]
    minimum_distance = min(distances) if distances else None
    minimum_on = min((row["visual_on_count"] for row in compact), default=None)
    return {
        "full_anchor": {
            "route_id": full_rows[0]["route_id"],
            "prediction": full_rows[0]["prediction"],
            "score": float(full_rows[0]["score"]),
            "correct": bool(full_rows[0]["result_correct"]),
            "generated_ids": full_rows[0]["generated_ids"],
        },
        "all_off_anchor": {
            "route_id": off_rows[0]["route_id"],
            "prediction": off_rows[0]["prediction"],
            "score": float(off_rows[0]["score"]),
            "correct": bool(off_rows[0]["result_correct"]),
            "generated_ids": off_rows[0]["generated_ids"],
        },
        "correcting_routes": compact,
        "correcting_route_count": len(compact),
        "nearest_correcting_route_distance": minimum_distance,
        "nearest_correcting_routes": [
            row for row in compact if row["hamming_distance_to_full"] == minimum_distance
        ],
        "minimum_correcting_visual_on_count": minimum_on,
    }


def build_rows(
    summaries: Iterable[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    indices: dict[str, dict[str, Any]],
    image_root: Path,
) -> tuple[list[dict[str, Any]], Counter]:
    rows: list[dict[str, Any]] = []
    taxonomy: Counter = Counter()
    for summary in summaries:
        dataset = str(summary["dataset"])
        if dataset not in {"gqa", "textvqa"}:
            continue
        uid = str(summary["uid"])
        cohort = classify_summary(summary)
        taxonomy[(dataset, cohort)] += 1
        if cohort not in {PRIMARY, CONTROL_NO_CORRECTION, CONTROL_VISION_REQUIRED}:
            continue
        source = sources[uid]
        index = indices[uid]
        raw_path = Path(index["record_path"])
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        image_path = image_root / dataset / Path(source["local_image_path"]).name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if image_path.stat().st_size != int(source["image_size_bytes"]):
            raise ValueError(f"image byte-size mismatch for {uid}")
        routes = None
        if cohort == PRIMARY:
            record = json.loads(raw_path.read_text(encoding="utf-8"))
            routes = route_metadata(record)
            if routes["full_anchor"]["correct"] or routes["all_off_anchor"]["correct"]:
                raise ValueError(f"invalid primary anchors for {uid}")
            if routes["correcting_route_count"] < 1:
                raise ValueError(f"primary record has no positive-vision correcting route: {uid}")
            if routes["correcting_route_count"] != int(summary["correcting_route_count"]):
                raise ValueError(f"correcting-route count drift for {uid}")
        rows.append(
            {
                "schema_version": "four_action_cohort_v1",
                "uid": uid,
                "dataset": dataset,
                "cohort": cohort,
                "sample_id": source["sample_id"],
                "image_id": source.get("source_asset_id") or source["image_group_id"],
                "image_group_id": source["image_group_id"],
                "image_path": str(image_path.resolve()),
                "image_size_bytes": int(source["image_size_bytes"]),
                "question": source["question"],
                "prompt": source["prompt"],
                "answer": source["answer"],
                "all_answer_norms": source.get("all_answer_norms"),
                "metric_name": source["metric_name"],
                "correctness_threshold": float(source["correctness_threshold"]),
                "max_new_tokens": int(source["max_new_tokens"]),
                "max_image_tokens": source.get("max_image_tokens"),
                "full_prediction": summary["current_all_on_prediction"],
                "full_score": float(summary["current_all_on_score"]),
                "full_correct": summary["current_all_on_status"] == "correct",
                "all_off_score": float(summary["all_off_score"]),
                "all_off_correct": bool(summary["all_off_correct"]),
                "visual_token_count": int(summary["actual_visual_tokens"]),
                "text_token_count": int(summary["actual_text_tokens"]),
                "full_prompt_token_count": int(summary["actual_full_prompt_tokens"]),
                "requested_simulations": int(summary["requested_simulations"]),
                "completed_simulations": int(summary["completed_simulations"]),
                "evaluated_route_count": int(summary["evaluated_route_count"]),
                "raw_record_path": str(raw_path),
                "raw_record_sha256": index["record_sha256"],
                "binary_routes": routes,
                "shard": shard_for(uid),
            }
        )
    return rows, taxonomy


def selection_ids(rows: list[dict[str, Any]], per_dataset: int) -> list[str]:
    output: list[str] = []
    for dataset in ("gqa", "textvqa"):
        population = [row for row in rows if row["dataset"] == dataset and row["cohort"] == PRIMARY]
        output.extend(evenly_spaced_ids(population, per_dataset))
    return output


def summarize(rows: list[dict[str, Any]], taxonomy: Counter) -> dict[str, Any]:
    grouped: dict[str, dict[str, int]] = defaultdict(dict)
    for (dataset, cohort), count in sorted(taxonomy.items()):
        grouped[dataset][cohort] = count
    primary = [row for row in rows if row["cohort"] == PRIMARY]
    return {
        "schema_version": "four_action_cohort_summary_v1",
        "taxonomy": dict(grouped),
        "emitted_rows": len(rows),
        "primary_rows": len(primary),
        "primary_ids": [row["uid"] for row in primary],
        "smoke_ids": selection_ids(rows, 4),
        "pilot_ids": selection_ids(rows, 28),
        "selection_rule": "per dataset, sort A+ by (visual_token_count, uid) and choose evenly spaced positions",
        "shard_count": 8,
    }
