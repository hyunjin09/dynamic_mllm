import subprocess

from experiments.monitor_four_action_multiplex_ramp import gpu_snapshot, throughput_assessment


def test_ramp_throughput_requires_material_improvement():
    accepted = throughput_assessment(
        first_count=1,
        final_count=65,
        steady_seconds=120.0,
        baseline_samples_per_minute=18.0,
        minimum_speedup=1.20,
    )
    assert accepted["samples_per_minute"] == 32.0
    assert accepted["speedup"] > 1.7
    assert accepted["passed"]

    slow = throughput_assessment(
        first_count=1,
        final_count=25,
        steady_seconds=80.0,
        baseline_samples_per_minute=18.0,
        minimum_speedup=1.20,
    )
    assert slow["samples_per_minute"] == 18.0
    assert not slow["passed"]


def test_ramp_throughput_rejects_missing_measurement_interval():
    result = throughput_assessment(
        first_count=1,
        final_count=1,
        steady_seconds=0.0,
        baseline_samples_per_minute=18.0,
        minimum_speedup=1.20,
    )
    assert result["samples_per_minute"] is None
    assert not result["passed"]


def test_cpu_partition_nvml_denial_is_diagnostic_not_a_monitor_crash(monkeypatch):
    def denied(*args, **kwargs):
        raise subprocess.CalledProcessError(6, ["nvidia-smi"], stderr="NVML denied")

    monkeypatch.setattr(subprocess, "run", denied)
    rows, error = gpu_snapshot()

    assert rows == []
    assert "exit 6" in error
