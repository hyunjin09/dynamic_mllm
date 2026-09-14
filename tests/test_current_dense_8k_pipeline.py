from __future__ import annotations

from pathlib import Path

import pytest

from dense_failure_stage1.current_dense_8k import (
    build_current_candidate_rows,
    summarize_population,
    validate_attempt_coverage,
)


def _row(uid: str, benchmark: str, bucket: str, suffix: str = ".jpg") -> dict:
    sample_id = uid.split(":", 1)[1]
    return {
        "uid": uid,
        "sample_id": sample_id,
        "benchmark": benchmark,
        "question": "question",
        "prompt": "question\nAnswer the question using a single word or phrase.",
        "answer": "answer",
        "all_answer_norms": ["answer"] * 10 if benchmark == "textvqa" else None,
        "source_bucket": bucket,
        "source_asset_id": f"asset:{sample_id}",
        "image_content_sha256": "a" * 64,
        "local_image_path": f"/old/{sample_id}{suffix}",
    }


def test_candidate_recovery_keeps_missing_images_for_worker_level_skip(tmp_path):
    rows = build_current_candidate_rows(
        [_row("gqa:a", "gqa", "complete_wrong")],
        image_root=tmp_path / "images",
    )

    assert len(rows) == 1
    assert rows[0]["uid"] == "gqa:a"
    assert rows[0]["historical_bucket"] == "wrong"
    assert rows[0]["image_group_id"] == f"sha256:{'a' * 64}"
    assert rows[0]["local_image_path"].endswith("gqa/wrong__a.jpg")
    assert rows[0]["image_present_at_preparation"] is False


def test_attempt_coverage_accepts_explicit_skips_but_not_lost_uids():
    candidates = [{"uid": "a"}, {"uid": "b"}, {"uid": "c"}]
    outputs = [{"uid": "a"}, {"uid": "b"}]
    skips = [{"uid": "c", "reason": "missing image"}]

    validate_attempt_coverage(candidates, outputs, skips)
    with pytest.raises(ValueError, match="coverage"):
        validate_attempt_coverage(candidates, outputs, [])
    with pytest.raises(ValueError, match="duplicate"):
        validate_attempt_coverage(candidates, outputs + [{"uid": "a"}], skips)


def test_population_summary_counts_binary_labels_and_skips():
    candidates = [
        {"uid": "gqa:a", "dataset": "gqa", "historical_bucket": "correct", "image_group_id": "i1"},
        {"uid": "gqa:b", "dataset": "gqa", "historical_bucket": "correct", "image_group_id": "i1"},
        {"uid": "textvqa:c", "dataset": "textvqa", "historical_bucket": "wrong", "image_group_id": "i2"},
    ]
    outputs = [
        {"uid": "gqa:a", "dataset": "gqa", "current_dense_correct": True, "historical_bucket": "correct", "image_group_id": "i1", "lmms_eval_per_sample_score": 1.0},
        {"uid": "gqa:b", "dataset": "gqa", "current_dense_correct": False, "historical_bucket": "correct", "image_group_id": "i1", "lmms_eval_per_sample_score": 0.0},
    ]
    skips = [{"uid": "textvqa:c", "dataset": "textvqa", "reason_code": "missing_image"}]

    summary = summarize_population(candidates, outputs, skips)

    assert summary["candidate_samples"] == 3
    assert summary["attempted"] == 3
    assert summary["successfully_completed"] == 2
    assert summary["skipped_or_failed"] == 1
    assert summary["by_dataset"]["gqa"] == {"correct": 1, "wrong": 1, "total": 2}
    assert summary["skip_reasons"] == {"missing_image": 1}
    assert summary["candidate_image_groups"] == 2
    assert summary["completed_image_groups"] == 1
    assert summary["historical_vs_current"] == {
        "correct_to_correct": 1,
        "correct_to_wrong": 1,
    }
