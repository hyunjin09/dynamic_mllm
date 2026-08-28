from __future__ import annotations

from hashlib import sha256
import json

import pytest

from experiments.prepare_sequential_dataset_priority_claims import (
    prepare_deferred_dataset_claims,
)


def test_prepare_claims_defers_only_selected_datasets(tmp_path):
    rows = [
        {"uid": "gqa:1", "dataset": "gqa"},
        {"uid": "textvqa:1", "dataset": "textvqa"},
        {"uid": "chartqa:1", "dataset": "chartqa"},
        {"uid": "wemath20_standard:1", "dataset": "wemath20_standard"},
        {"uid": "wemath2pro:1", "dataset": "wemath2pro"},
    ]

    summary = prepare_deferred_dataset_claims(
        rows,
        claim_root=tmp_path / "claims",
        deferred_datasets={"wemath20_standard", "wemath2pro"},
        claimant="deferred-dataset-order",
        resume=False,
    )

    expected_uids = {"wemath20_standard:1", "wemath2pro:1"}
    assert summary == {
        "total_samples": 5,
        "active_samples": 3,
        "deferred_samples": 2,
        "active_by_dataset": {"chartqa": 1, "gqa": 1, "textvqa": 1},
        "deferred_by_dataset": {"wemath20_standard": 1, "wemath2pro": 1},
        "claims_created": 2,
        "claims_verified": 0,
    }
    claim_files = sorted((tmp_path / "claims").glob("*.json"))
    assert {path.name for path in claim_files} == {
        f"{sha256(uid.encode()).hexdigest()}.json" for uid in expected_uids
    }
    assert {
        json.loads(path.read_text(encoding="utf-8"))["uid"] for path in claim_files
    } == expected_uids


def test_prepare_claims_resume_is_idempotent_and_verifies_payload(tmp_path):
    rows = [{"uid": "wemath20_standard:1", "dataset": "wemath20_standard"}]
    kwargs = {
        "claim_root": tmp_path / "claims",
        "deferred_datasets": {"wemath20_standard"},
        "claimant": "deferred-dataset-order",
    }
    prepare_deferred_dataset_claims(rows, resume=False, **kwargs)

    summary = prepare_deferred_dataset_claims(rows, resume=True, **kwargs)

    assert summary["claims_created"] == 0
    assert summary["claims_verified"] == 1


def test_prepare_claims_refuses_existing_or_mismatched_claims(tmp_path):
    rows = [{"uid": "wemath2pro:1", "dataset": "wemath2pro"}]
    claim_root = tmp_path / "claims"
    claim_root.mkdir()
    claim_path = claim_root / f"{sha256('wemath2pro:1'.encode()).hexdigest()}.json"
    claim_path.write_text(
        json.dumps({"uid": "different", "claimant": "rank-00"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError, match="without --resume"):
        prepare_deferred_dataset_claims(
            rows,
            claim_root=claim_root,
            deferred_datasets={"wemath2pro"},
            claimant="deferred-dataset-order",
            resume=False,
        )
    with pytest.raises(RuntimeError, match="unexpected existing claim"):
        prepare_deferred_dataset_claims(
            rows,
            claim_root=claim_root,
            deferred_datasets={"wemath2pro"},
            claimant="deferred-dataset-order",
            resume=True,
        )
