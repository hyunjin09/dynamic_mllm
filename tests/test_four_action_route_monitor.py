from __future__ import annotations

from pathlib import Path

import pytest

from experiments.monitor_route_conditioned_gpu import csv_open_contract


def test_gpu_monitor_open_contract_is_append_only_on_resume(tmp_path: Path):
    path = tmp_path / "gpu.csv"
    assert csv_open_contract(path, resume=False) == ("w", True)
    path.write_text("header\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        csv_open_contract(path, resume=False)
    assert csv_open_contract(path, resume=True) == ("a", False)
