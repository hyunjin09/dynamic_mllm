from __future__ import annotations

import pytest

from tools.research_analysis.four_action.sequential_label_jobs import (
    SEQUENTIAL_CONVERSION_CODE_PATHS,
    SequentialAtomicSampleQueue,
    build_sequential_execution_contract,
    mode_topology,
    select_sequential_smoke,
)


def test_frozen_mode_topology_uses_eight_smoke_and_sixteen_full_workers():
    assert mode_topology("smoke") == {
        "gpu_count": 8,
        "worker_count": 8,
        "workers_per_gpu": 1,
    }
    assert mode_topology("full") == {
        "gpu_count": 8,
        "worker_count": 16,
        "workers_per_gpu": 2,
    }


def test_sequential_contract_hashes_only_exact_policy_inputs(tmp_path):
    for relative in SEQUENTIAL_CONVERSION_CODE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    manifest_path = tmp_path / "manifest.jsonl"
    config_path.write_text("model: {}\n", encoding="utf-8")
    manifest_path.write_text('{"uid":"u"}\n', encoding="utf-8")

    contract = build_sequential_execution_contract(
        project_root=tmp_path,
        config_path=config_path,
        manifest_path=manifest_path,
        config={
            "model": {
                "snapshot_path": "models/qwen",
                "revision": "revision-1",
                "attention_implementation": "sdpa",
            },
            "seed": 7,
            "layer_count": 28,
            "processing_order": "early_to_late",
        },
        mode="smoke",
        git_commit="abc123",
        torch_version="2.6.0",
        transformers_version="5.3.0",
    )

    encoded = repr(contract).lower()
    assert "beam" not in encoded
    assert "route_cap" not in encoded
    assert "canonical" not in encoded
    assert contract["search_policy"] == "exact_sequential_verified_branching"
    assert contract["worker_topology"] == mode_topology("smoke")
    assert len(contract["contract_sha256"]) == 64


def test_smoke_selection_is_exactly_the_declared_eight_uids_with_required_coverage():
    datasets = (
        "gqa",
        "textvqa",
        "chartqa",
        "wemath20_standard",
        "wemath2pro",
    )
    rows = []
    for index in range(8):
        rows.append(
            {
                "uid": f"u{index}",
                "dataset": datasets[index % len(datasets)],
                "source_current_all_on_status": "wrong" if index in {0, 2, 5} else "correct",
                "source_positive_route_count": 2 if index == 6 else 1,
                "source_positive_routes": [
                    {
                        "source_all_off": index == 0,
                        "source_off_count": 28 if index == 0 else index,
                    }
                ],
            }
        )

    selected, coverage = select_sequential_smoke(rows, [f"u{x}" for x in range(8)])

    assert [row["uid"] for row in selected] == [f"u{x}" for x in range(8)]
    assert coverage["datasets"] == sorted(datasets)
    assert coverage["w2c_samples"] == 3
    assert coverage["c2c_samples"] == 5
    assert coverage["all_off_w2c_samples"] == 1
    assert coverage["multi_source_route_samples"] == 1


def test_smoke_selection_rejects_missing_semantic_coverage():
    rows = [
        {
            "uid": f"u{x}",
            "dataset": "gqa",
            "source_current_all_on_status": "correct",
            "source_positive_route_count": 1,
            "source_positive_routes": [{"source_all_off": False, "source_off_count": 0}],
        }
        for x in range(8)
    ]
    with pytest.raises(ValueError, match="coverage"):
        select_sequential_smoke(rows, [f"u{x}" for x in range(8)])


def test_dynamic_queue_claims_each_sample_once_without_beam_era_cost_policy(tmp_path):
    rows = [
        {"uid": "small-w2c", "estimated_conversion_cost": 4},
        {"uid": "large-c2c", "estimated_conversion_cost": 100},
        {"uid": "done", "estimated_conversion_cost": 1000},
    ]
    first = SequentialAtomicSampleQueue(
        rows,
        claim_root=tmp_path / "claims",
        completed_uids={"done"},
        claimant="rank-0",
    )
    second = SequentialAtomicSampleQueue(
        rows,
        claim_root=tmp_path / "claims",
        completed_uids={"done"},
        claimant="rank-1",
    )

    assert first.claim_next()["uid"] == "large-c2c"
    assert second.claim_next()["uid"] == "small-w2c"
    assert first.claim_next() is None
    assert second.claim_next() is None
