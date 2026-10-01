"""Final host composition agrees with the real stationary compiler contract."""

from __future__ import annotations

import numpy as np
import pytest
from generativeqc._stationary_composite_cuda import _canonical_gradient_sum
from generativeqc_compiler.method import resolve_method
from generativeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)


@pytest.fixture(params=["unpolarized", "polarized"])
def plan(request: pytest.FixtureRequest) -> StationaryGradientPlan:
    return StationaryGradientPlan(
        resolve_method("WB97M-V", spin=request.param),
        StationaryMeanField(SCF_POINT_MODEL),
    )


def test_real_plan_order_and_inputs_are_preserved(plan: StationaryGradientPlan) -> None:
    values = {name: np.zeros((2, 3), dtype=np.float64) for name in plan.source_names}
    for name, value in zip(plan.source_names[:3], (1e16, -1e16, 1.0), strict=True):
        values[name].fill(value)
    original = {name: value.copy() for name, value in values.items()}
    # Reversed completion order would produce zero instead of one in this case.
    shuffled = dict(reversed(tuple(values.items())))
    actual = _canonical_gradient_sum(plan, shuffled, 2)
    np.testing.assert_array_equal(actual, np.ones((2, 3)))
    np.testing.assert_array_equal(actual, plan.reduce_diagnostic(shuffled, atoms=2))
    for name, value in values.items():
        np.testing.assert_array_equal(value, original[name])
        assert not np.shares_memory(actual, value)


@pytest.mark.parametrize("invalid", ["missing", "extra"])
def test_real_plan_rejects_incomplete_or_unknown_sources(
    plan: StationaryGradientPlan, invalid: str
) -> None:
    values = {name: np.zeros((2, 3), dtype=np.float64) for name in plan.source_names}
    if invalid == "missing":
        values.pop(plan.source_names[0])
    else:
        values["unknown_source"] = np.zeros((2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="source coverage"):
        _canonical_gradient_sum(plan, values, 2)


def test_finite_components_cannot_publish_an_overflowed_sum(
    plan: StationaryGradientPlan,
) -> None:
    values = {name: np.zeros((2, 3), dtype=np.float64) for name in plan.source_names}
    for name in plan.source_names[:2]:
        values[name].fill(np.finfo(np.float64).max)
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="final gradient"):
        _canonical_gradient_sum(plan, values, 2)
