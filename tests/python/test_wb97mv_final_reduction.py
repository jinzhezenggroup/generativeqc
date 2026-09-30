"""Regression tests for WB97M-V final source composition."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from generativeqc._stationary_wb97mv_cuda import _canonical_gradient_sum

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "python/generativeqc/_stationary_wb97mv_cuda.py").read_text()


class FakePlan:
    source_names = ("a", "b", "c")

    def __init__(self) -> None:
        self.coverage = None

    def reduction_program(self, *, atoms: int, sources: object = None) -> object:
        names = tuple(sources)
        if set(names) != set(self.source_names):
            raise ValueError("incomplete source coverage")
        self.coverage = (atoms, names)
        return SimpleNamespace()


def test_canonical_host_sum_preserves_compiler_coverage_gate() -> None:
    plan = FakePlan()
    components = {
        "c": np.full((2, 3), 3.0),
        "a": np.full((2, 3), 1.0),
        "b": np.full((2, 3), 2.0),
    }
    actual = _canonical_gradient_sum(plan, components, 2)
    np.testing.assert_array_equal(actual, np.full((2, 3), 6.0))
    assert plan.coverage == (2, ("c", "a", "b"))


@pytest.mark.parametrize(
    "value,match",
    [
        (np.ones((2, 3), dtype=np.float32), "shape/dtype"),
        (np.ones((3, 2), dtype=np.float64), "shape/dtype"),
        (np.full((2, 3), np.nan, dtype=np.float64), "nonfinite"),
    ],
)
def test_canonical_host_sum_rejects_invalid_component(
    value: np.ndarray, match: str
) -> None:
    plan = FakePlan()
    components = {
        "a": value,
        "b": np.ones((2, 3), dtype=np.float64),
        "c": np.ones((2, 3), dtype=np.float64),
    }
    with pytest.raises(ValueError, match=match):
        _canonical_gradient_sum(plan, components, 2)


def test_wb97mv_does_not_reupload_host_components_for_final_add() -> None:
    assert "PreparedCuda(" not in SOURCE
    assert "compile_cuda(" not in SOURCE
    assert "plan_cuda(" not in SOURCE
    assert "self.reduction.execute(" not in SOURCE
    assert '"final_reduction_h2d_bytes": 0' in SOURCE
    assert '"final_reduction_d2h_bytes": 0' in SOURCE
