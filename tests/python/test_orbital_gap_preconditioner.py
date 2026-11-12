"""Guarded orbital-gap conditioning with unchanged true-residual equations."""

import numpy as np
import pytest
from generativeqc.response_problem import RotationLayout
from generativeqc.response_solver import DiagonalPreconditioner, GMRESOptions, solve
from generativeqc.spin_stationary import SpinRotationLayout
from test_response_krylov import _MatrixOperator


def test_gap_factory_uses_the_explicit_rotation_order_and_preserves_the_solution() -> (
    None
):
    """Noncanonical orbital ordering must not change occupied-major packing."""
    layout = RotationLayout((0, 2), (1, 3, 4))
    energies = np.array((-100.0, -1.0, -3.0, 0.2, 1.4))
    expected = np.array((99.0, 100.2, 101.4, 2.0, 3.2, 4.4))
    preconditioner = DiagonalPreconditioner.from_orbital_gaps(energies, layout)
    assert preconditioner is not None
    np.testing.assert_array_equal(preconditioner.diagonal, expected)
    matrix = np.diag(expected) + 0.03 * np.ones((layout.dimension,) * 2)
    operator = _MatrixOperator(matrix, layout.dimension)
    rhs = np.linspace(-0.2, 0.7, layout.dimension)
    options = GMRESOptions(atol=1e-13, rtol=1e-12, restart=6)
    direct = np.linalg.solve(matrix, rhs)
    conditioned = solve(operator, rhs, preconditioner=preconditioner, options=options)
    plain = solve(operator, rhs, options=options)
    assert conditioned.converged and plain.converged
    np.testing.assert_allclose(conditioned.solution, direct, atol=2e-13, rtol=2e-12)
    np.testing.assert_allclose(plain.solution, direct, atol=2e-13, rtol=2e-12)
    assert np.linalg.norm(
        matrix @ conditioned.solution - rhs
    ) <= options.atol + options.rtol * np.linalg.norm(rhs)


@pytest.mark.parametrize("gap", (0.0, 1e-16))
def test_singular_gap_declines_without_modifying_the_physical_denominator(
    gap: float,
) -> None:
    layout = RotationLayout((0,), (1, 2))
    energies = np.array((0.0, gap, 0.5))
    original = energies.copy()
    assert DiagonalPreconditioner.from_orbital_gaps(energies, layout) is None
    np.testing.assert_array_equal(energies, original)


@pytest.mark.parametrize(
    "energies", ((0.0, np.nan), (0.0, 1j), (0.0,), (-1e308, 1e308))
)
def test_gap_factory_rejects_invalid_energies_before_publishing(
    energies: tuple,
) -> None:
    with pytest.raises(ValueError, match="orbital"):
        DiagonalPreconditioner.from_orbital_gaps(energies, RotationLayout((0,), (1,)))


def test_gap_factory_preserves_negative_gaps() -> None:
    preconditioner = DiagonalPreconditioner.from_orbital_gaps(
        (0.5, -0.2), RotationLayout((0,), (1,))
    )
    assert preconditioner is not None
    np.testing.assert_array_equal(preconditioner.diagonal, (-0.7,))


@pytest.mark.parametrize("occupied", ((2, 1), (1, 0), (4, 1)))
def test_spin_gaps_preserve_coupled_packing_and_true_residual(occupied: tuple) -> None:
    """Unequal or empty spin blocks retain cross-spin matrix coupling."""
    layout = SpinRotationLayout(occupied, 4)
    energies = np.array(((-4, -2, 1, 3), (-3, -1, 0.5, 2)))
    expected = layout.pack(
        tuple(
            values[count:][None] - values[:count, None]
            for values, count in zip(energies, occupied, strict=True)
        )
    )
    preconditioner = DiagonalPreconditioner.from_spin_orbital_gaps(energies, occupied)
    assert preconditioner is not None
    np.testing.assert_array_equal(preconditioner.diagonal, expected)
    matrix = np.diag(expected) + 0.02 * np.ones((layout.dimension,) * 2)
    rhs = np.linspace(-0.3, 0.7, layout.dimension)
    options = GMRESOptions(atol=1e-13, rtol=1e-12, restart=layout.dimension)
    result = solve(
        _MatrixOperator(matrix, layout.dimension),
        rhs,
        preconditioner=preconditioner,
        options=options,
    )
    assert result.converged
    np.testing.assert_allclose(
        result.solution, np.linalg.solve(matrix, rhs), atol=2e-13, rtol=2e-12
    )
    assert np.linalg.norm(matrix @ result.solution - rhs) <= (
        options.atol + options.rtol * np.linalg.norm(rhs)
    )


@pytest.mark.parametrize("energies", (((0, 0), (0, 1)), ((0, 1e-15), (0, 2))))
def test_either_singular_spin_declines_without_clamping(energies: tuple) -> None:
    original = np.asarray(energies).copy()
    assert DiagonalPreconditioner.from_spin_orbital_gaps(energies, (1, 1)) is None
    np.testing.assert_array_equal(energies, original)


def test_spin_gap_scale_is_shared_across_spins_and_preserves_signed_gaps() -> None:
    assert (
        DiagonalPreconditioner.from_spin_orbital_gaps(((0, 1e-6), (0, 1e9)), (1, 1))
        is None
    )
    preconditioner = DiagonalPreconditioner.from_spin_orbital_gaps(
        ((0.5, -0.2), (-1.0, 2.0)), (1, 1)
    )
    assert preconditioner is not None
    np.testing.assert_allclose(preconditioner.diagonal, (-0.7, 3.0), atol=0, rtol=0)
    assert (
        DiagonalPreconditioner.from_spin_orbital_gaps(((0, 1), (0, 1)), (0, 2)) is None
    )


@pytest.mark.parametrize(
    ("energies", "occupied"),
    (
        ((0, 1), (1, 1)),
        (((0, np.nan), (0, 1)), (1, 1)),
        (((0, 1j), (0, 1)), (1, 1)),
        (((-1e308, 1e308), (0, 1)), (1, 1)),
        (((0, 1), (0, 1)), (1, 3)),
        (((0, 1), (0, 1)), (True, 1)),
    ),
)
def test_spin_gap_factory_rejects_invalid_data(
    energies: tuple, occupied: tuple
) -> None:
    with pytest.raises(ValueError, match="orbital|occupied"):
        DiagonalPreconditioner.from_spin_orbital_gaps(energies, occupied)
