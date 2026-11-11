"""Bounded validation must preserve the native ECP Hessian discretization."""

import types

import numpy as np
import pytest
from generativeqc.ecp_hessian import (
    HESSIAN_VALIDATION_POLAR_POINTS,
    HESSIAN_VALIDATION_RADIAL_POINTS,
    ECPSecondOrder,
)
from generativeqc_compiler.integral.ecp_policy import (
    COARSE_POLAR_POINTS,
    COARSE_RADIAL_POINTS,
    DERIVATIVE_ABS_TOLERANCE,
    REFINED_POLAR_POINTS,
    REFINED_RADIAL_POINTS,
)

GRIDS = {
    "coarse": (COARSE_RADIAL_POINTS, COARSE_POLAR_POINTS),
    "fine": (REFINED_RADIAL_POINTS, REFINED_POLAR_POINTS),
    "validation": (HESSIAN_VALIDATION_RADIAL_POINTS, HESSIAN_VALIDATION_POLAR_POINTS),
}


def owner_with_integrals(values: dict) -> tuple:
    """Supply grid-dependent integrals without depending on invocation order."""
    owner = object.__new__(ECPSecondOrder)
    owner._analytic_source = None
    owner.statistics = {"hvp_validation_refinements": 0}
    integrals = {GRIDS[name]: value for name, value in values.items()}

    def integrate(
        self: object,
        operation: str,
        *,
        radial: int = REFINED_RADIAL_POINTS,
        polar: int = REFINED_POLAR_POINTS,
        **options: object,
    ) -> np.ndarray:
        value = integrals[radial, polar]
        if isinstance(value, Exception):
            raise value
        return np.array(value, dtype=float, copy=True)

    owner._integrate = types.MethodType(integrate, owner)
    return owner, integrals


def apply(owner: ECPSecondOrder) -> dict:
    return owner.hvp(np.ones((2, 3)), np.eye(2), np.zeros((2, 2)))


def assert_native_output(result: dict, fine: np.ndarray) -> None:
    assert set(result) == {"ecp_local", "ecp_nonlocal"}
    for component, name in enumerate(("ecp_local", "ecp_nonlocal")):
        np.testing.assert_array_equal(result[name], fine[component])
        assert not result[name].flags.writeable


@pytest.mark.parametrize("error", (0.0, DERIVATIVE_ABS_TOLERANCE))
def test_original_gate_returns_native_output_without_requiring_validation(
    error: float,
) -> None:
    coarse = np.zeros((2, 2, 3))
    fine = coarse + error
    owner, _ = owner_with_integrals(
        {
            "coarse": coarse,
            "fine": fine,
            "validation": MemoryError("denser grid exceeds the available budget"),
        }
    )
    assert_native_output(apply(owner), fine)
    assert owner.statistics["hvp_validation_refinements"] == 0
    assert owner.statistics["hvp_coarse_error"] == error
    assert owner.statistics["hvp_validation_error"] is None


def test_finite_coarse_failure_validates_but_never_replaces_native_hvp() -> None:
    fine = np.arange(12, dtype=float).reshape(2, 2, 3)
    owner, _ = owner_with_integrals(
        {"coarse": fine + 1e-6, "fine": fine, "validation": fine + 1e-9}
    )
    assert_native_output(apply(owner), fine)
    assert owner.statistics["hvp_validation_refinements"] > 0
    assert owner.statistics["hvp_coarse_error"] > DERIVATIVE_ABS_TOLERANCE
    assert owner.statistics["hvp_validation_error"] < DERIVATIVE_ABS_TOLERANCE


def test_validation_equality_at_tolerance_remains_accepted() -> None:
    fine = np.zeros((2, 2, 3))
    owner, _ = owner_with_integrals(
        {
            "coarse": fine + 1e-6,
            "fine": fine,
            "validation": fine + DERIVATIVE_ABS_TOLERANCE,
        }
    )
    assert_native_output(apply(owner), fine)
    assert owner.statistics["hvp_validation_error"] == DERIVATIVE_ABS_TOLERANCE


@pytest.mark.parametrize("channel", (0, 1))
def test_unresolved_local_or_nonlocal_refinement_rejects(channel: int) -> None:
    fine = np.zeros((2, 2, 3))
    validation = fine.copy()
    validation[channel, 0, 0] = 2 * DERIVATIVE_ABS_TOLERANCE
    owner, _ = owner_with_integrals(
        {"coarse": fine + 1e-6, "fine": fine, "validation": validation}
    )
    with pytest.raises(ArithmeticError, match="quadrature convergence gate failed"):
        apply(owner)


def test_opposing_channel_errors_cannot_cancel_in_either_admission_gate() -> None:
    fine = np.zeros((2, 2, 3))
    coarse, validation = fine.copy(), fine.copy()
    coarse[:, 0, 0] = (1e-6, -1e-6)
    validation[:, 0, 0] = (
        2 * DERIVATIVE_ABS_TOLERANCE,
        -2 * DERIVATIVE_ABS_TOLERANCE,
    )
    np.testing.assert_array_equal((coarse - fine).sum(axis=0), 0)
    np.testing.assert_array_equal((validation - fine).sum(axis=0), 0)
    owner, _ = owner_with_integrals(
        {"coarse": coarse, "fine": fine, "validation": validation}
    )
    with pytest.raises(ArithmeticError, match="quadrature convergence gate failed"):
        apply(owner)


@pytest.mark.parametrize("grid", ("coarse", "fine", "validation"))
@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf))
def test_nonfinite_grids_never_admit_or_require_further_refinement(
    grid: str, bad: float
) -> None:
    fine = np.zeros((2, 2, 3))
    values = {"coarse": fine + 1e-6, "fine": fine.copy(), "validation": fine.copy()}
    values[grid][0, 0, 0] = bad
    if grid != "validation":
        values["validation"] = AssertionError("a nonfinite grid cannot be refined")
    owner, _ = owner_with_integrals(values)
    with pytest.raises(ArithmeticError, match="quadrature convergence gate failed"):
        apply(owner)


@pytest.mark.parametrize("stage", ("original", "validation"))
def test_finite_subtraction_overflow_rejects(stage: str) -> None:
    largest = np.full((2, 2, 3), np.finfo(float).max)
    if stage == "original":
        values = {
            "coarse": largest,
            "fine": -largest,
            "validation": AssertionError("an overflowed comparison cannot be refined"),
        }
    else:
        values = {
            "coarse": np.zeros_like(largest),
            "fine": largest,
            "validation": -largest,
        }
    owner, _ = owner_with_integrals(values)
    with (
        np.errstate(over="ignore"),
        pytest.raises(ArithmeticError, match="quadrature convergence gate failed"),
    ):
        apply(owner)


def test_validation_budget_failure_propagates_and_same_owner_recovers() -> None:
    fine = np.zeros((2, 2, 3))
    owner, integrals = owner_with_integrals(
        {"coarse": fine + 1e-6, "fine": fine, "validation": fine + 1e-9}
    )
    assert_native_output(apply(owner), fine)
    assert owner.statistics["hvp_validation_error"] is not None

    integrals[GRIDS["validation"]] = MemoryError(
        "ECP radial-layer jets exceed grid_budget_bytes"
    )
    with pytest.raises(MemoryError, match="grid_budget_bytes"):
        apply(owner)
    assert owner.statistics["hvp_coarse_error"] > DERIVATIVE_ABS_TOLERANCE
    assert owner.statistics["hvp_validation_error"] is None

    # A failed action must not poison the owner or require the unavailable dense
    # grid when a subsequent action satisfies the original acceptance policy.
    integrals[GRIDS["coarse"]] = fine
    assert_native_output(apply(owner), fine)
    assert owner.statistics["hvp_coarse_error"] == 0
    assert owner.statistics["hvp_validation_error"] is None
