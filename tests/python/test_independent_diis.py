"""Check oracle DIIS algebra and real cold SCFs, not heavy-domain admission."""

import numpy as np
import pytest
from independent_diis import normalized_diis_factory, normalized_error_coefficients
from test_global_hybrid_ecp_basis import canonical_records as _canonical_records
from test_global_hybrid_ecp_forces import (
    IndependentSCFConvergenceError,
    independent_solver,
)

canonical_records = _canonical_records


@pytest.mark.parametrize("scale", (1e-240, 1e-24, 1e-12, 1.0, 1e12, 1e240))
def test_common_gram_scale_preserves_the_affine_minimizer(scale: float) -> None:
    """An independently solved well-conditioned KKT system fixes the answer."""
    errors = np.array([[0.2, 0.8, -0.3], [0.6, -0.1, 0.4], [-0.4, 0.1, 0.5]])
    gram = errors @ errors.T
    augmented = np.ones((4, 4))
    augmented[0, 0] = 0
    augmented[1:, 1:] = gram
    expected = np.linalg.solve(augmented, np.array([1.0, 0.0, 0.0, 0.0]))[1:]
    actual = normalized_error_coefficients(gram * scale)
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
    assert abs(actual.sum() - 1) < 1e-13


@pytest.mark.parametrize("scale", (1e-24, 1.0, 1e24))
def test_rank_deficient_histories_remain_bounded_and_minimal(scale: float) -> None:
    gram = np.ones((4, 4)) * scale
    coefficients = normalized_error_coefficients(gram)
    np.testing.assert_allclose(coefficients, np.full(4, 0.25), atol=2e-14, rtol=0)
    np.testing.assert_allclose(
        normalized_error_coefficients(np.zeros((4, 4))),
        coefficients,
        atol=2e-14,
        rtol=0,
    )


@pytest.mark.parametrize("scale", (1e-24, 1.0, 1e24))
def test_redundant_opposing_errors_reach_the_exact_zero_residual(scale: float) -> None:
    errors = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [1.0, 0.0]])
    coefficients = normalized_error_coefficients((errors @ errors.T) * scale)
    np.testing.assert_allclose(coefficients @ errors, np.zeros(2), atol=1e-13, rtol=0)
    assert abs(coefficients.sum() - 1) < 1e-13


@pytest.mark.parametrize("scale", (1e-12, 1.0, 1e12))
def test_actual_history_extrapolation_uses_one_common_error_scale(scale: float) -> None:
    pytest.importorskip("pyscf")
    from pyscf.lib.diis import DIIS

    mixer = normalized_diis_factory(None)
    vectors = np.array([[1.0, -2.0], [3.0, 4.0], [-2.0, 1.0]])
    errors = np.array([[0.2, 0.8, -0.3], [0.6, -0.1, 0.4], [-0.4, 0.1, 0.5]])
    gram = errors @ errors.T
    augmented = np.ones((4, 4))
    augmented[0, 0] = 0
    augmented[1:, 1:] = gram
    coefficients = np.linalg.solve(augmented, np.array([1.0, 0.0, 0.0, 0.0]))[1:]
    for vector, error in zip(vectors, errors, strict=True):
        actual = DIIS.update(mixer, vector, xerr=error * scale)
    np.testing.assert_allclose(actual, coefficients @ vectors, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "bad",
    (
        np.empty((0, 0)),
        np.ones((2, 3)),
        np.ones((65, 65)),
        np.array([[1.0, np.nan], [np.nan, 1.0]]),
        np.array([[1.0, np.inf], [np.inf, 1.0]]),
        np.array([[1.0, 0.2], [0.3, 1.0]]),
        np.array([[1.0, 2.0], [2.0, 1.0]]),
        np.eye(2, dtype=complex),
    ),
)
def test_invalid_gram_cannot_publish_coefficients(bad: object) -> None:
    with pytest.raises(ValueError):
        normalized_error_coefficients(bad)


def test_near_parallel_errors_cannot_publish_unbounded_cancellation() -> None:
    errors = np.array([[1.0], [1.0 + 4e-7]])
    with pytest.raises(ArithmeticError):
        normalized_error_coefficients(errors @ errors.T)


@pytest.mark.parametrize("multiplicity,charge", ((1, 0), (2, 1)))
def test_normalized_factory_keeps_real_cold_state_and_strict_physical_gate(
    canonical_records: tuple,
    multiplicity: int,
    charge: int,
) -> None:
    pytest.importorskip("pyscf")
    _, raw, _ = canonical_records
    solver = independent_solver(
        raw,
        [("H", (0.0, 0.0, -0.7)), ("H", (0.1, 0.0, 0.7))],
        charge,
        multiplicity,
        max_iterations=240,
        orbital_gradient_tolerance=1e-11,
        final_diagonalization=False,
        diis_factory=normalized_diis_factory,
    )
    assert solver.converged
    assert np.linalg.norm(solver.get_grad(solver.mo_coeff, solver.mo_occ)) < 1e-11
    assert solver.diis.normalization_records
    density = solver.make_rdm1()
    electrons = float(np.einsum("...ij,ji->...", density, solver.get_ovlp()).sum())
    assert abs(electrons - (2 - charge)) < 1e-10
    default = independent_solver(
        raw,
        [("H", (0.0, 0.0, -0.7)), ("H", (0.1, 0.0, 0.7))],
        charge,
        multiplicity,
    )
    assert abs(solver.e_tot - default.e_tot) < 1e-11
    np.testing.assert_allclose(density, default.make_rdm1(), atol=1e-9, rtol=0)


def test_normalized_factory_cannot_certify_an_unconverged_state(
    canonical_records: tuple,
) -> None:
    pytest.importorskip("pyscf")
    _, raw, _ = canonical_records
    with pytest.raises(IndependentSCFConvergenceError) as failure:
        independent_solver(
            raw,
            [("H", (0.0, 0.0, -0.7)), ("H", (0.1, 0.0, 0.7))],
            0,
            1,
            max_iterations=1,
            orbital_gradient_tolerance=1e-11,
            final_diagonalization=False,
            diis_factory=normalized_diis_factory,
        )
    assert not failure.value.solver.converged
