"""Scaling diagnostics must not promote experimental work or hide regressions."""

import os

import pytest

from benchmarks.pbe0_fast_scaling import (
    CONFLICTS,
    CONTROLS,
    experimental_mode,
    summarize,
)


def test_experimental_overrides_restore_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in CONFLICTS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    with (
        pytest.raises(RuntimeError, match="injected"),
        experimental_mode("experimental_xc") as selected,
    ):
        assert selected == "auto"
        assert tuple(os.environ[name] for name in CONTROLS) == ("3", "1")
        raise RuntimeError("injected")
    assert all(name not in os.environ for name in CONTROLS)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    with experimental_mode("fast") as selected:
        assert selected == "fast"
        assert all(name not in os.environ for name in CONTROLS)


def test_existing_override_is_rejected_not_silently_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(CONTROLS[0], "2")
    with pytest.raises(ValueError, match="unset"), experimental_mode("experimental_xc"):
        pytest.fail("must not execute")
    assert os.environ[CONTROLS[0]] == "2"


def test_independent_gates_include_first_calls_and_keep_slow_samples() -> None:
    rows = [
        {"mode": mode, "phase": "measured", "seconds": seconds, "oracle_gate": True}
        for mode, seconds in (("fp64", 3.0), ("experimental_xc", 3.2), ("fast", 6.0))
    ]
    modes = ("fp64", "experimental_xc", "fast")
    report = summarize(rows, modes)
    assert report["speedup_vs_fp64"]["experimental_xc"] == pytest.approx(0.9375)
    assert report["speedup_vs_fp64"]["fast"] == 0.5
    rows.append({**rows[0], "phase": "first-call", "seconds": 100})
    assert summarize(rows, modes) == report
    rows[-1]["oracle_gate"] = False
    with pytest.raises(ValueError, match="rejected"):
        summarize(rows, modes)
