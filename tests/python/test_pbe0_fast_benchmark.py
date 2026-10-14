"""Fast measurements must not hide current-FP64 performance or disable guards."""

import os
from typing import Any, NoReturn

import pytest

from benchmarks import pbe0_fast as benchmark


def test_summary_compares_with_optimized_current_fp64() -> None:
    """The same exact optimization belongs to ordinary FP64, not only Fast."""
    records = [
        {"mode": mode, "seconds": seconds, "oracle_gate": True, "phase": "measured"}
        for mode, seconds in zip(benchmark.MODES, (3.0, 3.2), strict=True)
    ]
    summary = benchmark.summarize(records)
    assert "speedup_vs_no_reuse" not in summary
    assert summary["speedup_vs_current_fp64"]["fast"] == pytest.approx(0.9375)
    records.append({**records[0], "seconds": 100, "phase": "first-call"})
    assert benchmark.summarize(records) == summary
    records[-1]["oracle_gate"] = False
    with pytest.raises(ValueError, match="rejected"):
        benchmark.summarize(records)


def test_benchmark_never_changes_scheduler_controls_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(benchmark.CONTROL, raising=False)

    def fail(*args: Any) -> NoReturn:
        assert benchmark.CONTROL not in os.environ
        raise RuntimeError("native failure")

    monkeypatch.setattr(benchmark, "native_sample", fail)
    with pytest.raises(RuntimeError, match="native failure"):
        benchmark.sample("fp64", {}, "case")
    assert benchmark.CONTROL not in os.environ
    monkeypatch.setenv(benchmark.CONTROL, "1")
    with pytest.raises(ValueError, match="unset"):
        benchmark.sample("fp64", {}, "case")
    assert os.environ[benchmark.CONTROL] == "1"


def test_fast_benchmark_uses_real_facade_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(benchmark.CONTROL, raising=False)

    def native(mode: str, options: dict, case: str) -> dict:
        assert mode == "fast" and benchmark.CONTROL not in os.environ
        return {"seconds": 1.0}

    monkeypatch.setattr(benchmark, "native_sample", native)
    assert benchmark.sample("fast", {}, "case")["mode"] == "fast"


def test_mixed_force_gate_does_not_relax_strict_comparator() -> None:
    protocol = {"energy_gate_Eh": 1e-7, "force_gate_Eh_per_bohr": 1e-6}
    errors = {
        "energy_Eh": 1e-9,
        "max_force_Eh_per_bohr": 2e-6,
        "rms_force_Eh_per_bohr": 1e-6,
    }
    assert benchmark.oracle_gate("fast", errors, protocol)
    assert not benchmark.oracle_gate("fp64", errors, protocol)
    errors["rms_force_Eh_per_bohr"] = 4e-6
    assert not benchmark.oracle_gate("fast", errors, protocol)
