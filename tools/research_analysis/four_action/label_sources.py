from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _binary_mask(value: Sequence[int | bool], *, layer_count: int) -> list[int]:
    mask = [int(item) for item in value]
    if len(mask) != layer_count or any(item not in {0, 1} for item in mask):
        raise ValueError(f"binary route must contain exactly {layer_count} 0/1 actions")
    return mask


def _route_row(
    row: Mapping[str, Any],
    *,
    uid: str,
    mask_field: str,
    layer_count: int,
) -> dict[str, Any]:
    route_id = str(row["route_id"])
    mask = _binary_mask(row[mask_field], layer_count=layer_count)
    return {
        "source_binary_route_id": f"{uid}::{route_id}",
        "route_id": route_id,
        "mask": mask,
        "mask_key": "".join(map(str, mask)),
        "source_score": None if row.get("score") is None else float(row["score"]),
        "source_reward": None if row.get("reward") is None else float(row["reward"]),
        "source_off_count": len(mask) - sum(mask),
        "source_all_off": not any(mask),
    }


def _common_record(
    sample: Mapping[str, Any],
    *,
    dataset: str,
    split: str,
    image_path: Path,
    routes: list[dict[str, Any]],
    source_artifact: str,
) -> dict[str, Any]:
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    expected_size = sample.get("image_size_bytes")
    if expected_size is not None and image_path.stat().st_size != int(expected_size):
        raise ValueError(f"image byte-size mismatch for {sample['uid']}")
    return {
        "schema_version": "four_action_label_source_v1",
        "uid": str(sample["uid"]),
        "dataset": dataset,
        "benchmark": str(sample.get("benchmark") or dataset),
        "sample_id": str(sample["sample_id"]),
        "source_split": split,
        "image_id": str(sample.get("source_asset_id") or sample.get("image_group_id") or ""),
        "image_group_id": str(sample.get("image_group_id") or image_path),
        "image_path": str(image_path.resolve()),
        "image_content_sha256": sample.get("image_content_sha256"),
        "image_size_bytes": image_path.stat().st_size,
        "question": str(sample["question"]),
        "prompt": str(sample["prompt"]),
        "answer": str(sample["answer"]),
        "all_answer_norms": sample.get("all_answer_norms"),
        "metric_name": str(sample["metric_name"]),
        "correctness_threshold": float(sample["correctness_threshold"]),
        "max_new_tokens": int(sample["max_new_tokens"]),
        "max_image_tokens": sample.get("max_image_tokens"),
        "source_current_all_on_status": sample.get("current_all_on_status"),
        "source_current_all_on_prediction": sample.get("current_all_on_prediction"),
        "source_current_all_on_score": sample.get("current_all_on_score"),
        "source_artifact": source_artifact,
        "source_positive_route_count": len(routes),
        "source_positive_routes": routes,
        "estimated_conversion_cost": sum(1 + route["source_off_count"] for route in routes),
    }


def normalize_vqa_record(
    predictor: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    image_root: Path,
    layer_count: int = 28,
    source_artifact: str,
) -> dict[str, Any] | None:
    """Join one frozen VQA predictor row to its canonical source metadata."""
    uid = str(predictor["uid"])
    if uid != str(source["uid"]):
        raise ValueError("predictor/source UID mismatch")
    dataset = str(predictor["benchmark"]).lower()
    if dataset not in {"gqa", "textvqa", "chartqa"}:
        raise ValueError(f"unsupported VQA dataset: {dataset}")
    selected = list(predictor.get("valid_routes") or [])
    if len(selected) != int(predictor.get("selected_valid_route_count", len(selected))):
        raise ValueError(f"selected route count mismatch for {uid}")
    routes = []
    seen = set()
    for row in selected:
        if float(row.get("reward", 0.0)) <= 0.0:
            raise ValueError(f"non-positive selected route for {uid}")
        normalized = _route_row(
            row,
            uid=uid,
            mask_field="mask",
            layer_count=layer_count,
        )
        if normalized["source_binary_route_id"] in seen:
            raise ValueError(f"duplicate selected route for {uid}")
        seen.add(normalized["source_binary_route_id"])
        routes.append(normalized)
    if not routes:
        return None
    image_path = image_root / dataset / Path(str(source["local_image_path"])).name
    sample = dict(source)
    sample.update(
        {
            "current_all_on_status": predictor.get("current_all_on_status")
            or source.get("historical_all_on_status"),
            "current_all_on_prediction": predictor.get("current_all_on_prediction")
            or source.get("historical_all_on_prediction"),
            "current_all_on_score": predictor.get("current_all_on_score")
            if predictor.get("current_all_on_score") is not None
            else source.get("historical_all_on_score"),
        }
    )
    return _common_record(
        sample,
        dataset=dataset,
        split=str(predictor["split"]),
        image_path=image_path,
        routes=routes,
        source_artifact=source_artifact,
    )


def normalize_math_record(
    raw: Mapping[str, Any],
    *,
    image_path: Path,
    record_path: Path,
    record_sha256: str,
    layer_count: int = 28,
) -> dict[str, Any] | None:
    """Normalize the positive routes in one authoritative math cache record."""
    sample = dict(raw["sample"])
    uid = str(sample["uid"])
    benchmark = str(sample["benchmark"])
    dataset = {
        "wemath20_standard": "wemath20_standard",
        "wemath2pro": "wemath2pro",
    }.get(benchmark)
    if dataset is None:
        raise ValueError(f"unsupported math benchmark: {benchmark}")
    successful_ids = [str(value) for value in raw.get("successful_route_ids") or []]
    if len(successful_ids) != len(set(successful_ids)):
        raise ValueError(f"duplicate successful route IDs for {uid}")
    candidates = {str(row["route_id"]): row for row in raw.get("candidate_executions") or []}
    missing = sorted(set(successful_ids) - set(candidates))
    if missing:
        raise ValueError(f"missing successful candidates for {uid}: {missing[:3]}")
    routes = []
    for route_id in successful_ids:
        row = candidates[route_id]
        if row.get("result_correct") is not True:
            raise ValueError(f"successful route must be evaluator-correct for {uid}: {route_id}")
        routes.append(
            _route_row(
                row,
                uid=uid,
                mask_field="visual_on_mask",
                layer_count=layer_count,
            )
        )
    if not routes:
        return None
    normalized = _common_record(
        sample,
        dataset=dataset,
        split=str(sample.get("source_split") or dataset),
        image_path=image_path,
        routes=routes,
        source_artifact=str(record_path.resolve()),
    )
    normalized["source_record_sha256"] = record_sha256
    normalized["source_dataset_version"] = raw.get("dataset_version")
    return normalized
