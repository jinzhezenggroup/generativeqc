"""Regression tests for WB97M-V final source composition."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from generativeqc._stationary_composite_cuda import _canonical_gradient_sum
from generativeqc_compiler.method import resolve_method
from generativeqc_compiler.method.stationary_gradient import (
    StationaryGradientPlan,
    StationaryMeanField,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "python/generativeqc/_stationary_composite_cuda.py").read_text()


class FakePlan:
    source_names = ("a", "b", "c")

    def __init__(self) -> None:
        self.coverage = None

    def validate_source_coverage(self, *, sources: object = None) -> tuple[str, ...]:
        names = tuple(sources)
        if set(names) != set(self.source_names):
            raise ValueError("incomplete source coverage")
        self.coverage = names
        return self.source_names


def test_canonical_host_sum_preserves_compiler_coverage_gate() -> None:
    plan = FakePlan()
    components = {
        "c": np.full((2, 3), 3.0),
        "a": np.full((2, 3), 1.0),
        "b": np.full((2, 3), 2.0),
    }
    actual = _canonical_gradient_sum(plan, components, 2)
    np.testing.assert_array_equal(actual, np.full((2, 3), 6.0))
    assert plan.coverage == ("c", "a", "b")


@pytest.mark.parametrize("changed", (None, "missing", "unknown"))
def test_host_sum_uses_real_compiler_source_coverage(changed: str | None) -> None:
    """Reject incomplete compositions without requiring unused reduction IR."""
    plan = StationaryGradientPlan(
        resolve_method("WB97M-V"),
        StationaryMeanField("libxc-7.0/work-mgga-v1/smooth-lr-a1.35-order16"),
    )
    components = {
        name: np.full((2, 3), float(index + 1))
        for index, name in enumerate(plan.source_names)
    }
    if changed == "missing":
        components.pop(plan.source_names[0])
    elif changed == "unknown":
        components["unplanned"] = np.zeros((2, 3))
    if changed is not None:
        with pytest.raises(ValueError, match="source coverage"):
            _canonical_gradient_sum(plan, components, 2)
    else:
        count = len(plan.source_names)
        np.testing.assert_array_equal(
            _canonical_gradient_sum(plan, components, 2),
            np.full((2, 3), count * (count + 1) / 2),
        )


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
