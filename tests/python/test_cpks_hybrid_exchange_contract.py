"""CPU-only contract checks for global-hybrid closed-shell CPKS K response.

These are independent tiny AO algebra tests, not native PBE0/B3LYP or
molecular Hessian qualification. No CUDA runtime is required.
"""

from dataclasses import replace
from fractions import Fraction
import typing

import numpy as np
import pytest
from generativeqc.response_operator import (
    CPKSResponseOperator,
    cpks_operator_identity,
)
from generativeqc.response_problem import ResponseUnsupported

from tools.generativeqc_posthf.fixtures import (
    fixture_snapshot,
    load_fixture,
)
from tools.generativeqc_response import DenseAOResponseBackend


class LinearXCKernel:
    """Synthetic symmetric XC action with an exact identity-bound reference."""

    def __init__(self, reference: typing.Any, coefficient: float = 0.31) -> None:
        self.basis_identity = reference.basis_hash
        self.grid_identity = reference.grid_identity
        self.functional_identity = reference.functional_identity
        self.identity = f"linear-hybrid-cpks-xc-{coefficient}"
        self.coefficient = coefficient
        self.host_workspace_bytes = 0

    def apply(self, density: typing.Any) -> np.ndarray:
        return self.coefficient * np.asarray(density)

    def apply_transpose(self, density: typing.Any) -> np.ndarray:
        return self.apply(density)


def synthetic_problem() -> tuple[typing.Any, DenseAOResponseBackend, LinearXCKernel, np.ndarray]:
    """Return a restricted KS fixture and a tiny *raw* J/K oracle provider."""
    meta, arrays = load_fixture("h2")
    rhf = fixture_snapshot(meta, arrays)
    reference = replace(
        rhf,
        algorithm="KS",
        functional_identity="synthetic-global-hybrid-physical-state",
        grid_identity="synthetic-fixed-grid",
        hf_backend="independent-cpu-jk-oracle",
    )
    n = reference.nmo
    core = np.arange(1, n * n + 1, dtype=np.float64).reshape(n, n)
    core = (core + core.T) / (2.0 * n * n)
    eri = (
        np.einsum("pq,rs->pqrs", core, core)
        + np.einsum("pr,qs->pqrs", core, core)
        + np.einsum("ps,qr->pqrs", core, core)
    )
    backend = DenseAOResponseBackend(eri)
    kernel = LinearXCKernel(reference)
    return reference, backend, kernel, eri


@pytest.mark.parametrize("fraction", [Fraction(1, 4), Fraction(1, 5), 0.37])
def test_hybrid_cpks_raw_k_coefficient_and_directional_differences(
    fraction: typing.Any,
) -> None:
    reference, backend, kernel, eri = synthetic_problem()
    # Only the test's complete raw J/K oracle is explicitly exchange-qualified.
    backend.supports_full_range_exchange = True
    problem = CPKSResponseOperator.build_problem(
        reference, backend, kernel, exact_exchange_fraction=fraction
    )
    operator = CPKSResponseOperator(
        problem, backend, kernel, exact_exchange_fraction=fraction
    )
    physical_a = float(fraction)
    assert operator.exact_exchange_fraction == physical_a
    assert operator.exchange_fraction == physical_a / 2.0
    assert problem.operator_identity == cpks_operator_identity(
        backend, kernel, exact_exchange_fraction=fraction
    )

    direction = np.array([[0.3, -0.2], [-0.2, -0.1]], dtype=np.float64)
    base = np.array([[0.7, 0.15], [0.15, 0.2]], dtype=np.float64)

    def independent_fock(density: np.ndarray) -> np.ndarray:
        j = np.einsum("pqrs,rs->pq", eri, density)
        k = np.einsum("prqs,rs->pq", eri, density)
        return j - 0.5 * physical_a * k + kernel.coefficient * density

    expected = independent_fock(direction)
    np.testing.assert_allclose(operator.induced_fock(direction), expected, atol=1e-13)
    for step in (1e-3, 1e-4, 1e-5):
        numerical = (
            independent_fock(base + step * direction)
            - independent_fock(base - step * direction)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            operator.induced_fock(direction), numerical, atol=3e-11, rtol=3e-11
        )

    vector = np.linspace(0.2, 0.2 * problem.dimension, problem.dimension)
    x = problem.layout.validate_vector(vector)
    c = reference.coefficients
    delta_ao = c @ problem.layout.density_matrix(x) @ c.T
    fock_mo = c.T @ independent_fock(delta_ao) @ c
    occ = problem.layout.occupied
    vir = problem.layout.virtual
    gaps = (
        reference.orbital_energies[list(vir)][None, :]
        - (reference.orbital_energies[list(occ)][:, None])
    )
    expected_action = (
        gaps * x.reshape(len(occ), len(vir)) + fock_mo[np.ix_(vir, occ)].T
    ).reshape(-1)
    np.testing.assert_allclose(operator.apply(vector), expected_action, atol=1e-12)


def test_hybrid_cpks_refuses_j_only_or_unqualified_provider() -> None:
    reference, backend, kernel, _ = synthetic_problem()
    with pytest.raises(ResponseUnsupported, match="full-range K provider"):
        CPKSResponseOperator.build_problem(
            reference, backend, kernel, exact_exchange_fraction=Fraction(1, 4)
        )
    semilocal = CPKSResponseOperator.build_problem(reference, backend, kernel)
    assert semilocal.operator_identity == cpks_operator_identity(backend, kernel)
    assert CPKSResponseOperator(semilocal, backend, kernel).exchange_fraction == 0.0


def test_hybrid_cpks_fraction_is_part_of_problem_identity() -> None:
    reference, backend, kernel, _ = synthetic_problem()
    backend.supports_full_range_exchange = True
    pbe0 = CPKSResponseOperator.build_problem(
        reference, backend, kernel, exact_exchange_fraction=Fraction(1, 4)
    )
    b3lyp = CPKSResponseOperator.build_problem(
        reference, backend, kernel, exact_exchange_fraction=Fraction(1, 5)
    )
    assert pbe0.operator_identity != b3lyp.operator_identity
    with pytest.raises(ValueError, match="operator_identity"):
        CPKSResponseOperator(
            pbe0, backend, kernel, exact_exchange_fraction=Fraction(1, 5)
        )
    # An admitted provider cannot withdraw K mid-solve without failing closed.
    live = CPKSResponseOperator(
        pbe0, backend, kernel, exact_exchange_fraction=Fraction(1, 4)
    )
    backend.supports_full_range_exchange = False
    with pytest.raises(ResponseUnsupported, match="full-range K provider"):
        live.induced_fock(np.eye(reference.nmo))


@pytest.mark.parametrize("invalid", [True, "0.25", float("nan"), -0.1, 1.1])
def test_hybrid_cpks_rejects_invalid_exchange_fraction(invalid: typing.Any) -> None:
    reference, backend, kernel, _ = synthetic_problem()
    backend.supports_full_range_exchange = True
    with pytest.raises((TypeError, ValueError), match="exact-exchange fraction"):
        CPKSResponseOperator.build_problem(
            reference, backend, kernel, exact_exchange_fraction=invalid
        )
