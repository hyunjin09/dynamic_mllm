#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import experiments.run_sequential_four_action_label_conversion as base_runner
from tools.research_analysis.four_action.sequential_label_jobs import file_sha256


THREE_REPLICA_TOPOLOGY = {
    "gpu_count": 8,
    "worker_count": 24,
    "workers_per_gpu": 3,
}
BASE_MODE_TOPOLOGY = base_runner.mode_topology
BASE_BUILD_EXECUTION_CONTRACT = base_runner.build_sequential_execution_contract


def three_replica_mode_topology(mode: str) -> dict[str, int]:
    if mode == "full":
        return dict(THREE_REPLICA_TOPOLOGY)
    return BASE_MODE_TOPOLOGY(mode)


def build_three_replica_execution_contract(
    *, launcher_path: Path | None = None, **kwargs: Any
) -> dict[str, Any]:
    """Extend the frozen scientific contract with truthful 24-worker provenance."""
    contract = deepcopy(BASE_BUILD_EXECUTION_CONTRACT(**kwargs))
    contract.pop("contract_sha256", None)
    contract["worker_topology"] = dict(THREE_REPLICA_TOPOLOGY)

    project_root = Path(kwargs["project_root"]).resolve()
    launcher = (
        Path(__file__).resolve()
        if launcher_path is None
        else Path(launcher_path).resolve()
    )
    relative_launcher = str(launcher.relative_to(project_root))
    contract["code_sha256"][relative_launcher] = file_sha256(launcher)
    encoded = json.dumps(
        contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    contract["contract_sha256"] = sha256(encoded.encode()).hexdigest()
    return contract


def main() -> int:
    base_runner.mode_topology = three_replica_mode_topology
    base_runner.build_sequential_execution_contract = (
        build_three_replica_execution_contract
    )
    return base_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
