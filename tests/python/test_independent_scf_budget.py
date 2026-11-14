"""Check independent-oracle resource policy without qualifying a heavy endpoint."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from test_global_hybrid_ecp_basis import canonical_records as _canonical_records
from test_global_hybrid_ecp_forces import (
    IndependentSCFConvergenceError,
    independent_solver,
)

canonical_records = _canonical_records


@pytest.mark.parametrize(
    "max_iterations,gradient_tolerance,final_diagonalization",
    ((240, 1e-10, True), (600, 1e-11, True), (600, 1e-11, False)),
)
def test_independent_allowance_preserves_scf_tolerances(
    canonical_records: tuple,
    max_iterations: int,
    gradient_tolerance: float,
    final_diagonalization: bool,
) -> None:
    pytest.importorskip("pyscf")
    _, raw, _ = canonical_records
    solver = independent_solver(
        raw,
        [("H", (0.0, 0.0, -0.7)), ("H", (0.1, 0.0, 0.7))],
        0,
        1,
        max_iterations=max_iterations,
        orbital_gradient_tolerance=gradient_tolerance,
        final_diagonalization=final_diagonalization,
    )
    assert solver.converged
    assert solver.max_cycle == max_iterations
    assert solver.conv_tol == 1e-13
    assert solver.conv_tol_grad == gradient_tolerance
    if not final_diagonalization:
        assert (
            np.linalg.norm(solver.get_grad(solver.mo_coeff, solver.mo_occ))
            < gradient_tolerance
        )


def test_unconverged_oracle_retains_the_actual_failed_state(
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
        )
    solver = failure.value.solver
    assert not solver.converged
    assert solver.max_cycle == 1
    assert solver.make_rdm1().shape == solver.get_ovlp().shape
    assert solver.mo_coeff is not None and solver.mo_occ is not None


@pytest.mark.parametrize(
    "charge,multiplicity,counts",
    ((0, 1, (2, 2)), (1, 2, (2, 1)), (0, 3, (3, 1)), (1, 4, (3, 0)), (0, 5, (4, 0))),
)
def test_independent_spin_populations_follow_the_requested_multiplicity(
    canonical_records: tuple, charge: int, multiplicity: int, counts: tuple[int, int]
) -> None:
    """Require unrestricted orbitals, not merely a two-component spin density.

    PySCF's RKS factory returns ROKS for open shells. Its density has alpha/beta
    components but its shared orbital matrix cannot qualify UKS response.
    """
    pytest.importorskip("pyscf")
    _, raw, _ = canonical_records
    solver = independent_solver(
        raw,
        [("H", (0.3 * atom, 0.1 * atom * atom, 1.6 * atom)) for atom in range(4)],
        charge,
        multiplicity,
        orbital_gradient_tolerance=1e-10,
    )
    overlap = solver.get_ovlp()
    orbital_shape = (overlap.shape[0], overlap.shape[0])
    if multiplicity != 1:
        orbital_shape = (2, *orbital_shape)
    assert np.asarray(solver.mo_coeff).shape == orbital_shape
    density = np.asarray(solver.make_rdm1())
    if multiplicity == 1:
        density = np.stack((0.5 * density, 0.5 * density))
    assert density.shape == (2, overlap.shape[0], overlap.shape[1])
    np.testing.assert_allclose(
        np.einsum("sij,ji->s", density, overlap), counts, atol=1e-10, rtol=0
    )
    assert np.linalg.norm(solver.get_grad(solver.mo_coeff, solver.mo_occ)) < 2e-9


@pytest.mark.parametrize("residual", (1e-6, np.nan))
def test_no_extra_step_cannot_accept_a_bad_physical_residual(
    canonical_records: tuple, monkeypatch: pytest.MonkeyPatch, residual: float
) -> None:
    """A success flag alone must not admit even this synthetic failed state."""
    dft = pytest.importorskip("pyscf.dft")
    solver = SimpleNamespace(
        grids=SimpleNamespace(),
        converged=True,
        mo_coeff=np.eye(1),
        mo_occ=np.array([2.0]),
        kernel=lambda: None,
        get_grad=lambda *args: np.array([residual]),
    )
    monkeypatch.setattr(dft, "RKS", lambda *args: solver)
    _, raw, _ = canonical_records
    with pytest.raises(IndependentSCFConvergenceError) as failure:
        independent_solver(
            raw,
            [("H", (0.0, 0.0, -0.7)), ("H", (0.1, 0.0, 0.7))],
            0,
            1,
            max_iterations=600,
            orbital_gradient_tolerance=1e-11,
            final_diagonalization=False,
        )
    assert failure.value.solver is solver
    assert solver.converged


@pytest.mark.parametrize("order", (2, 4))
def test_independent_stencil_has_its_declared_polynomial_accuracy(order: int) -> None:
    from test_native_canonical_heavy_hessian import _gradient_column, _gradient_offsets

    step = 0.03
    linear = np.array([[0.3, -0.2, 0.7]])
    cubic = np.array([[0.8, 1.7, -0.9]])
    samples = {}
    for offset in _gradient_offsets(order):
        position = offset * step
        samples[offset] = (
            1.2
            + linear * position
            + 0.5 * position**2
            + cubic * position**3
            + 0.1 * position**4
        )
    expected = linear + (cubic * step**2 if order == 2 else 0)
    np.testing.assert_allclose(
        _gradient_column(samples, step, order), expected, atol=2e-14, rtol=0
    )


@pytest.mark.parametrize("change", ("missing", "shape", "complex", "nonfinite", "step"))
def test_independent_stencil_rejects_incomplete_or_invalid_data(change: str) -> None:
    from test_native_canonical_heavy_hessian import _gradient_column

    samples = {offset: np.ones((2, 3)) for offset in (-2, -1, 1, 2)}
    step = 0.003
    if change == "missing":
        del samples[2]
    elif change == "shape":
        samples[2] = np.ones((1, 3))
    elif change == "complex":
        samples[2] = samples[2] + 1j
    elif change == "nonfinite":
        samples[2][0, 0] = np.nan
    else:
        step = 0
    with pytest.raises(ValueError):
        _gradient_column(samples, step, 4)
