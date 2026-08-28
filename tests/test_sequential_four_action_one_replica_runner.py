from __future__ import annotations

from hashlib import sha256
import json

from experiments.run_sequential_four_action_label_conversion_one_replica import (
    ONE_REPLICA_TOPOLOGY,
    build_one_replica_execution_contract,
    one_replica_mode_topology,
)


def test_one_replica_topology_changes_only_full_mode():
    assert one_replica_mode_topology("smoke") == {
        "gpu_count": 8,
        "worker_count": 8,
        "workers_per_gpu": 1,
    }
    assert one_replica_mode_topology("full") == ONE_REPLICA_TOPOLOGY == {
        "gpu_count": 8,
        "worker_count": 8,
        "workers_per_gpu": 1,
    }


def test_one_replica_contract_records_actual_topology_and_launcher(monkeypatch, tmp_path):
    launcher = tmp_path / "experiments" / "one.py"
    launcher.parent.mkdir()
    launcher.write_text("one-replica-launcher\n", encoding="utf-8")
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
        "experiments.run_sequential_four_action_label_conversion_one_replica."
        "BASE_BUILD_EXECUTION_CONTRACT",
        lambda **_kwargs: base_contract,
    )

    contract = build_one_replica_execution_contract(
        project_root=tmp_path,
        launcher_path=launcher,
    )

    assert contract["worker_topology"] == ONE_REPLICA_TOPOLOGY
    assert base_contract["worker_topology"]["worker_count"] == 16
    assert contract["code_sha256"]["experiments/one.py"] == sha256(
        launcher.read_bytes()
    ).hexdigest()
    unsigned = dict(contract)
    observed_hash = unsigned.pop("contract_sha256")
    encoded = json.dumps(
        unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    assert observed_hash == sha256(encoded.encode()).hexdigest()
