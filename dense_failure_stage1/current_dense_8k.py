"""Data and audit helpers for the permissive current-dense Stage-1 run."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence


STAGE1_DATASETS = ("gqa", "chartqa", "textvqa")


def build_current_candidate_rows(
    portable_rows: Iterable[Mapping], *, image_root: Path
) -> list[dict]:
    """Recover all three-dataset candidates without dropping bad image rows."""

    candidates = []
    seen = set()
    for source_index, source in enumerate(portable_rows):
        dataset = str(source.get("benchmark", "")).lower()
        if dataset not in STAGE1_DATASETS:
            continue
        uid = str(source.get("uid", ""))
        if not uid:
            raise ValueError(f"candidate at source index {source_index} has no UID")
        if uid in seen:
            raise ValueError(f"duplicate candidate UID: {uid}")
        seen.add(uid)
        bucket = {
            "complete_correct": "correct",
            "complete_wrong": "wrong",
        }.get(str(source.get("source_bucket")))
        if bucket is None:
            raise ValueError(f"{uid} has unsupported historical bucket")
        suffix = Path(str(source.get("local_image_path", ""))).suffix
        if not suffix:
            raise ValueError(f"{uid} has no recoverable image suffix")
        sample_id = str(source.get("sample_id", ""))
        image_path = (
            image_root / dataset / f"{bucket}__{sample_id}{suffix}"
        ).resolve()
        image_sha256 = str(source.get("image_content_sha256", ""))
        if len(image_sha256) != 64:
            raise ValueError(f"{uid} has no valid image content SHA-256")
        answers = source.get("all_answer_norms")
        if dataset == "textvqa" and not answers:
            raise ValueError(f"{uid} has no TextVQA reference answers")
        required = ("question", "prompt", "answer")
        for field in required:
            if source.get(field) is None or str(source[field]) == "":
                raise ValueError(f"{uid} has empty required field {field}")
        candidates.append(
            {
                "schema_version": "current_dense_8k_candidate_v1",
                "uid": uid,
                "sample_id": sample_id,
                "dataset": dataset,
                "question": str(source["question"]),
                "prompt": str(source["prompt"]),
                "answer": str(source["answer"]),
                "all_answer_norms": answers,
                "local_image_path": str(image_path),
                "image_content_sha256": image_sha256,
                "image_group_id": f"sha256:{image_sha256}",
                "image_identifier": str(source.get("source_asset_id") or image_sha256),
                "historical_bucket": bucket,
                "source_manifest_index": source_index,
                "max_new_tokens": int(source.get("max_new_tokens", 16)),
                "image_present_at_preparation": image_path.is_file(),
            }
        )
    candidates.sort(key=lambda row: row["uid"])
    return candidates


def validate_attempt_coverage(
    candidates: Sequence[Mapping], outputs: Sequence[Mapping], skips: Sequence[Mapping]
) -> None:
    expected = [str(row["uid"]) for row in candidates]
    completed = [str(row["uid"]) for row in outputs]
    skipped = [str(row["uid"]) for row in skips]
    observed = completed + skipped
    if len(observed) != len(set(observed)):
        raise ValueError("duplicate UID across completed/skipped records")
    if set(observed) != set(expected) or len(observed) != len(expected):
        missing = sorted(set(expected) - set(observed))[:10]
        extra = sorted(set(observed) - set(expected))[:10]
        raise ValueError(f"attempt coverage differs: missing={missing} extra={extra}")


def summarize_population(
    candidates: Sequence[Mapping], outputs: Sequence[Mapping], skips: Sequence[Mapping]
) -> dict:
    validate_attempt_coverage(candidates, outputs, skips)
    by_dataset = {
        dataset: {"correct": 0, "wrong": 0, "total": 0}
        for dataset in STAGE1_DATASETS
    }
    for row in outputs:
        dataset = str(row["dataset"])
        correct = bool(row["current_dense_correct"])
        by_dataset[dataset]["correct" if correct else "wrong"] += 1
        by_dataset[dataset]["total"] += 1
    overall_correct = sum(row["correct"] for row in by_dataset.values())
    overall_wrong = sum(row["wrong"] for row in by_dataset.values())
    historical_vs_current = Counter(
        (
            str(row["historical_bucket"]),
            "correct" if bool(row["current_dense_correct"]) else "wrong",
        )
        for row in outputs
    )
    return {
        "candidate_samples": len(candidates),
        "attempted": len(outputs) + len(skips),
        "successfully_completed": len(outputs),
        "skipped_or_failed": len(skips),
        "complete_28_layer_features": len(outputs),
        "candidate_image_groups": len(
            {str(row["image_group_id"]) for row in candidates}
        ),
        "completed_image_groups": len(
            {str(row["image_group_id"]) for row in outputs}
        ),
        "by_dataset": by_dataset,
        "overall": {
            "correct": overall_correct,
            "wrong": overall_wrong,
            "total": overall_correct + overall_wrong,
        },
        "skip_reasons": dict(
            sorted(Counter(str(row["reason_code"]) for row in skips).items())
        ),
        "historical_vs_current": {
            f"{historical}_to_{current}": count
            for (historical, current), count in sorted(
                historical_vs_current.items()
            )
        },
    }
