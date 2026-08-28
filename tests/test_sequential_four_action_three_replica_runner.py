from __future__ import annotations

from hashlib import sha256
import json

from experiments.run_sequential_four_action_label_conversion_three_replicas import (
    THREE_REPLICA_TOPOLOGY,
    build_three_replica_execution_contract,
    three_replica_mode_topology,
)


def test_three_replica_topology_changes_only_full_mode():
    assert three_replica_mode_topology("smoke") == {
        "gpu_count": 8,
        "worker_count": 8,
        "workers_per_gpu": 1,
    }
    assert three_replica_mode_topology("full") == THREE_REPLICA_TOPOLOGY == {
        "gpu_count": 8,
        "worker_count": 24,
        "workers_per_gpu": 3,
    }


def test_three_replica_contract_records_actual_topology_and_launcher(monkeypatch, tmp_path):
    launcher = tmp_path / "experiments" / "three.py"
    launcher.parent.mkdir()
    launcher.write_text("three-replica-launcher\n", encoding="utf-8")
    base_contract = {
        "schema_version": "exact_sequential_four_action_execution_contract_v1",
        "worker_topology": {
            "gpu_count": 8,
            "worker_count": 16,
            "workers_per_gpu": 2,
        },
        "code_sha256": {"scientific.py": "abc"},
        "contract_sha256": "old",
    }
    monkeypatch.setattr(
        "experiments.run_sequential_four_action_label_conversion_three_replicas."
        "BASE_BUILD_EXECUTION_CONTRACT",
        lambda **_kwargs: base_contract,
    )

    contract = build_three_replica_execution_contract(
        project_root=tmp_path,
        launcher_path=launcher,
    )

    assert contract["worker_topology"] == THREE_REPLICA_TOPOLOGY
    assert base_contract["worker_topology"]["worker_count"] == 16
    assert contract["code_sha256"]["experiments/three.py"] == sha256(
        launcher.read_bytes()
    ).hexdigest()
    unsigned = dict(contract)
    observed_hash = unsigned.pop("contract_sha256")
    encoded = json.dumps(
        unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    assert observed_hash == sha256(encoded.encode()).hexdigest()
