"""Host-side negative controls for shared Hessian publication/response gates."""

import numpy as np
import pytest
from generativeqc._cuda_hessian_response import resident_gmres
from generativeqc.ecp_hessian import ECPSecondOrder
from generativeqc.response_problem import ResponseSolveError
from generativeqc.response_solver import GMRESOptions


@pytest.mark.parametrize("value", (np.nan, np.inf, 1e308))
def test_resident_solver_rejects_nonfinite_or_overflowed_rhs_norm(value: float) -> None:
    # Exercise the same array-namespace algorithm without pretending this is a
    # real-device test. An infinite tolerance must never accept the zero guess.
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="norm.*nonfinite|norm.*overflows"),
    ):
        resident_gmres(
            np,
            lambda vector: 2 * vector,
            np.full(4, value),
            np.full(4, 2),
            GMRESOptions(),
        )


@pytest.mark.parametrize("scale", (1.0, 1e-270, 1e200))
def test_resident_solver_accepts_only_true_residuals(scale: float) -> None:
    matrix = np.array([[2.0, 0.3], [-0.2, 1.5]])
    rhs = np.array([0.3, -0.4]) * scale
    result = resident_gmres(
        np, lambda vector: matrix @ vector, rhs, np.diag(matrix), GMRESOptions()
    )
    np.testing.assert_allclose(
        result.solution / scale, np.linalg.solve(matrix, rhs / scale), atol=2e-14
    )
    assert np.linalg.norm((rhs - matrix @ result.solution) / scale) < 1e-12
    with pytest.raises(ResponseSolveError):
        resident_gmres(
            np,
            lambda vector: matrix @ vector,
            rhs,
            np.diag(matrix),
            GMRESOptions(restart=1, max_iterations=1, rtol=1e-14),
        )


@pytest.mark.parametrize("grid", ("coarse", "fine"))
@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_ecp_quadrature_rejects_either_nonfinite_grid(grid: str, value: float) -> None:
    owner = ECPSecondOrder.__new__(ECPSecondOrder)
    coarse, fine = np.zeros((2, 1, 3)), np.zeros((2, 1, 3))
    (coarse if grid == "coarse" else fine)[0, 0, 0] = value
    results = iter((coarse, fine))
    owner._integrate = lambda *args, **kwargs: next(results)
    with pytest.raises(ArithmeticError, match="quadrature convergence gate"):
        owner.hvp(np.zeros((1, 3)), np.zeros((1, 1)), np.zeros((1, 1)))


@pytest.mark.parametrize("difference", (0.0, 1.0))
def test_ecp_quadrature_finite_convergence_control(difference: float) -> None:
    owner = ECPSecondOrder.__new__(ECPSecondOrder)
    coarse, fine = np.zeros((2, 1, 3)), np.full((2, 1, 3), difference)
    results = iter((coarse, fine))
    owner._integrate = lambda *args, **kwargs: next(results)
    args = np.zeros((1, 3)), np.zeros((1, 1)), np.zeros((1, 1))
    if difference:
        with pytest.raises(ArithmeticError, match="quadrature convergence gate"):
            owner.hvp(*args)
    else:
        result = owner.hvp(*args)
        for value in result.values():
            np.testing.assert_array_equal(value, 0.0)
            assert not value.flags.writeable
