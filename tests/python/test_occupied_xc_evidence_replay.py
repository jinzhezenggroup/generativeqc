"""Diagnostic profile roundoff must not disable independent endpoint replay."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/occupied-xc-cold-20261023/reproduce.py"
)


def verify(actual: dict, expected: dict) -> None:
    spec = importlib.util.spec_from_file_location("occupied_xc_reproduce", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify_profile(actual, expected)


def test_profile_accepts_cross_python_summation_roundoff() -> None:
    actual = {"xc_calls": 11, "occupied_generations": [0, 2], "gpu_ms": 13669.07018}
    expected = dict(actual, gpu_ms=13669.070179999999)
    verify(actual, expected)
    assert actual["gpu_ms"] == 13669.07018
    assert expected["gpu_ms"] == 13669.070179999999


@pytest.mark.parametrize("value", [13669.071, float("nan"), float("inf")])
def test_profile_rejects_changed_or_nonfinite_timing(value: float) -> None:
    expected = {"xc_calls": 11, "occupied_generations": [0, 2], "gpu_ms": 13669.07018}
    with pytest.raises(AssertionError):
        verify(dict(expected, gpu_ms=value), expected)


@pytest.mark.parametrize("change", [{"xc_calls": 10}, {"occupied_generations": [0, 3]}])
def test_profile_still_requires_exact_counts_and_generations(change: dict) -> None:
    expected = {"xc_calls": 11, "occupied_generations": [0, 2], "gpu_ms": 13669.07018}
    with pytest.raises(AssertionError):
        verify(dict(expected, **change), expected)
