"""Independent AH algebra, failure isolation and cold-SCF certification checks."""

import numpy as np
import pytest
from independent_newton import (
    equilibrated_augmented_inputs,
    equilibrated_newton_reference,
)


@pytest.mark.parametrize("scales", ((1.0, 1.0), (1e-12, 2e-8), (-1e-12, 1e12)))
def test_augmented_congruence_preserves_physical_eigenstep(scales: tuple) -> None:
    pytest.importorskip("pyscf")
    from pyscf import lib
    from pyscf.soscf import ciah

    physical = np.array([[0.0, 0.02, -0.03], [0.02, 1.1, 0.17], [-0.03, 0.17, 0.6]])
    transform = np.diag([1.0, *scales])
    matrix = transform @ physical @ transform
    overlap = transform @ transform
    vectors = np.diag(scales)
    expected_values, expected_vectors = np.linalg.eigh(physical)
    expected_step = expected_vectors[1:, 0] / expected_vectors[0, 0]
    logger = lib.logger.Logger(None, lib.logger.QUIET)
    with equilibrated_newton_reference() as records:
        actual, value, coefficients, _selected, _spectrum = ciah._regular_step(
            matrix, overlap, list(vectors), 1e-14, logger
        )
    np.testing.assert_allclose(actual, expected_step, atol=1e-12, rtol=1e-12)
    assert abs(value - expected_values[0]) < 1e-12
    np.testing.assert_allclose(
        coefficients[1:] @ vectors / coefficients[0], actual, atol=1e-12, rtol=1e-12
    )
    assert records and all(record["physical_step_norm"] > 0 for record in records)


def test_equilibration_retains_true_linear_dependence() -> None:
    overlap = np.array([[1.0, 0.0, 0.0], [0.0, 1e-24, 2e-20], [0.0, 2e-20, 4e-16]])
    _matrix, normalized, _vectors, _scales = equilibrated_augmented_inputs(
        np.zeros((3, 3)), overlap, [[1e-12, 0.0], [2e-8, 0.0]]
    )
    np.testing.assert_allclose(np.diag(normalized), np.ones(3), atol=1e-14, rtol=0)
    assert np.linalg.eigvalsh(normalized).min() < 1e-14


@pytest.mark.parametrize(
    "invalid", ("complex", "shape", "nan", "zero", "affine", "asymmetric", "indefinite")
)
def test_invalid_projected_inputs_fail_closed(invalid: str) -> None:
    matrix, overlap, vectors = np.eye(3), np.eye(3), np.eye(2)
    if invalid == "complex":
        matrix = matrix.astype(complex)
    elif invalid == "shape":
        vectors = np.eye(3)
    elif invalid == "nan":
        vectors[0, 0] = np.nan
    elif invalid == "zero":
        overlap[1, 1] = 0
    elif invalid == "affine":
        overlap[0, 0] = 2
    elif invalid == "asymmetric":
        matrix[0, 1] = 1
    elif invalid == "indefinite":
        overlap[1, 2] = overlap[2, 1] = 2
    with pytest.raises(ValueError):
        equilibrated_augmented_inputs(matrix, overlap, vectors)


def test_reference_binding_restores_after_failure() -> None:
    pytest.importorskip("pyscf")
    from pyscf.soscf import ciah

    original = ciah._regular_step
    original_davidson = ciah.davidson_cc
    with (
        pytest.raises(RuntimeError, match="injected"),
        equilibrated_newton_reference(),
    ):
        raise RuntimeError("injected independent failure")
    assert ciah._regular_step is original
    assert ciah.davidson_cc is original_davidson


def test_zero_initial_trial_restarts_from_unchanged_physical_residual() -> None:
    pytest.importorskip("pyscf")
    from pyscf.soscf import ciah

    hessian = np.array([[0.7, 0.1], [0.1, 1.2]])
    gradient = np.array([2e-8, -3e-8])
    augmented = np.block(
        [[np.zeros((1, 1)), gradient[None]], [gradient[:, None], hessian]]
    )
    values, vectors = np.linalg.eigh(augmented)
    expected = vectors[1:, 0] / vectors[0, 0]
    with equilibrated_newton_reference() as records:
        iterations = list(
            ciah.davidson_cc(
                lambda vector: hessian @ vector,
                lambda: gradient,
                lambda vector, shift: vector / (np.diag(hessian) - shift),
                np.zeros(2),
                tol=1e-20,
                max_cycle=2,
                lindep=1e-14,
                verbose=0,
            )
        )
    np.testing.assert_allclose(iterations[-1][3], expected, atol=1e-15, rtol=1e-12)
    assert abs(iterations[-1][2] - values[0]) < (
        16 * np.finfo(float).eps * np.linalg.norm(augmented)
    )
    residual = hessian @ iterations[-1][3] + gradient
    residual -= iterations[-1][2] * iterations[-1][3]
    assert np.linalg.norm(residual) < 1e-15
    assert records[-1]["zero_initial_trial_restarts"] == 1
    np.testing.assert_array_equal(gradient, [2e-8, -3e-8])


@pytest.mark.parametrize("gradient", (np.zeros(2), np.array([np.nan, 1.0])))
def test_zero_initial_trial_without_finite_residual_fails(gradient: np.ndarray) -> None:
    pytest.importorskip("pyscf")
    from pyscf.soscf import ciah

    original = ciah.davidson_cc
    with (
        pytest.raises(ArithmeticError, match="no nonzero physical residual"),
        equilibrated_newton_reference(),
    ):
        list(
            ciah.davidson_cc(
                lambda vector: vector,
                lambda: gradient,
                lambda vector, shift: vector,
                np.zeros(2),
            )
        )
    assert ciah.davidson_cc is original


@pytest.mark.parametrize("charge,spin", ((0, 0), (1, 1)))
def test_actual_cold_reference_keeps_strict_physical_gate(
    charge: int, spin: int
) -> None:
    pytest.importorskip("pyscf")
    from pyscf import dft, gto, lib

    lib.num_threads(1)
    molecule = gto.M(
        atom=[("O", (0, 0, 0)), ("H", (0, -1.4, 1.1)), ("H", (0, 1.4, 1.1))],
        unit="Bohr",
        basis="sto-3g",
        charge=charge,
        spin=spin,
        verbose=0,
    )
    base = dft.UKS(molecule) if spin else dft.RKS(molecule)
    base.xc = "0.81*GGA_C_LYP+0.72*GGA_X_B88+0.19*LDA_C_VWN_RPA+0.08*LDA_X+0.2*HF"
    base.grids.level = 0
    base.small_rho_cutoff = 0
    solver = base.newton()
    solver.conv_tol, solver.conv_tol_grad, solver.max_cycle = 1e-13, 1e-11, 60
    solver.canonicalization = False
    with equilibrated_newton_reference():
        solver.kernel()
    assert solver.converged
    actual = np.linalg.norm(solver.get_grad(solver.mo_coeff, solver.mo_occ))
    assert actual < 1e-11
