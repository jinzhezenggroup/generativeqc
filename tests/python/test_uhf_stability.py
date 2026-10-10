"""Independent rotated-energy and lifecycle gates for internal real UHF."""

import os
import typing
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.linalg import expm

from tools.generativeqc_response import (
    DenseAOResponseBackend,
    UHFReferenceSnapshot,
    UHFResponseOperator,
)
from tools.generativeqc_response.stability import (
    StabilityOptions,
    StabilityUnqualified,
    diagnose_uhf_stability,
)


def _energy(
    hcore: np.ndarray, eri: np.ndarray, ca: np.ndarray, cb: np.ndarray, na: int, nb: int
) -> float:
    da, db = ca[:, :na] @ ca[:, :na].T, cb[:, :nb] @ cb[:, :nb].T
    total = da + db
    return float(
        np.einsum("pq,pq", hcore, total)
        + 0.5 * np.einsum("pqrs,pq,rs", eri, total, total)
        - 0.5 * sum(np.einsum("prqs,pq,rs", eri, d, d) for d in (da, db))
    )


def _operator(
    na: int = 2, nb: int = 1, *, interacting: bool = True, degenerate: bool = False
) -> tuple:
    n = 4
    rng = np.random.default_rng(1826)
    factors = rng.normal(scale=0.12, size=(n, n, n + 1))
    factors = (factors + factors.swapaxes(0, 1)) / 2
    eri = np.einsum("pqP,rsP->pqrs", factors, factors)
    # Orbital-sign symmetry makes both self-consistent Focks diagonal while
    # retaining mixed-spin OV couplings. This is a physical common hcore.
    for indices in np.ndindex(eri.shape):
        if any(indices.count(i) % 2 for i in range(n)):
            eri[indices] = 0
    if not interacting:
        eri.fill(0)
    hcore = np.diag(
        [-2.0, -1.0, 1.0, 2.0] if not degenerate else [-1.0, -1.0, 1.0, 1.0]
    )
    ds = [np.diag([1.0] * count + [0.0] * (n - count)) for count in (na, nb)]
    j = np.einsum("pqrs,rs->pq", eri, ds[0] + ds[1])
    focks = [hcore + j - np.einsum("prqs,rs->pq", eri, d) for d in ds]
    reference = UHFReferenceSnapshot(
        np.eye(n),
        hcore,
        *focks,
        np.eye(n),
        np.eye(n),
        *(np.diag(f) for f in focks),
        *(np.diag(d) for d in ds),
        _energy(hcore, eri, np.eye(n), np.eye(n), na, nb),
        0.0,
        "synthetic-geometry",
        "synthetic-basis",
        "synthetic-generation",
    )
    backend = DenseAOResponseBackend(eri)
    return UHFResponseOperator(
        UHFResponseOperator.build_problem(reference, backend), backend
    ), eri


def _rotated_energy(
    operator: UHFResponseOperator, eri: np.ndarray, vector: np.ndarray
) -> float:
    reference = operator.problem.reference
    coefficients = []
    offset = 0
    for spin in ("alpha", "beta"):
        n, no = reference.nbf, reference.nocc(spin)
        block = vector[offset : offset + no * (n - no)].reshape(no, n - no)
        offset += block.size
        generator = np.zeros((n, n))
        # Independent of layout.generator_matrix and density-response code.
        generator[no:, :no] = block.T
        generator[:no, no:] = -block
        coefficients.append(
            getattr(reference, f"coefficients_{spin}") @ expm(generator)
        )
    return _energy(
        reference.hcore,
        eri,
        *coefficients,
        reference.nocc("alpha"),
        reference.nocc("beta"),
    )


def _energy_hessian(
    operator: UHFResponseOperator, eri: np.ndarray, step: float = 2e-4
) -> np.ndarray:
    n = operator.dimension
    eye = np.eye(n) * step
    origin = _rotated_energy(operator, eri, np.zeros(n))
    hessian = np.empty((n, n))
    for i in range(n):
        hessian[i, i] = (
            _rotated_energy(operator, eri, eye[i])
            + _rotated_energy(operator, eri, -eye[i])
            - 2 * origin
        ) / step**2
        for j in range(i):
            hessian[i, j] = hessian[j, i] = sum(
                a * b * _rotated_energy(operator, eri, a * eye[i] + b * eye[j])
                for a in (-1, 1)
                for b in (-1, 1)
            ) / (4 * step**2)
    return hessian


def test_full_hessian_matches_independent_energy_and_cross_spin() -> None:
    operator, eri = _operator()
    result = diagnose_uhf_stability(operator)
    oracle = _energy_hessian(operator, eri)
    eye = np.eye(operator.dimension)
    actual = np.column_stack([2 * operator.apply(v) for v in eye])
    np.testing.assert_allclose(actual, oracle, atol=3e-7, rtol=2e-6)
    assert np.linalg.norm(oracle[:4, 4:]) > 1e-3
    np.testing.assert_allclose(
        result.eigenvalues, np.linalg.eigvalsh(oracle), atol=3e-7
    )
    assert result.status == "STABLE" and result.direction is None
    assert result.response_actions == 2 * operator.dimension
    assert result.reference_evaluations == 1 and result.elapsed_seconds > 0
    result.assert_current(operator)
    for wrong in (-actual, actual / 2, actual * 2):
        assert np.linalg.norm(wrong - oracle) > 1


@pytest.mark.parametrize("na,nb", [(1, 0), (4, 4), (2, 1)])
def test_empty_unequal_and_internal_degeneracy(na: int, nb: int) -> None:
    operator, _ = _operator(na, nb, interacting=False, degenerate=True)
    result = diagnose_uhf_stability(operator)
    assert result.status == ("EMPTY" if na == nb == 4 else "NEAR_SINGULAR")
    assert result.direction is None
    assert result.dimension == na * (4 - na) + nb * (4 - nb)


def test_spin_permutation_preserves_spectrum() -> None:
    operator, eri = _operator()
    r = operator.problem.reference
    swapped = replace(
        r,
        fock_alpha=r.fock_beta,
        fock_beta=r.fock_alpha,
        orbital_energies_alpha=r.orbital_energies_beta,
        orbital_energies_beta=r.orbital_energies_alpha,
        occupations_alpha=r.occupations_beta,
        occupations_beta=r.occupations_alpha,
    )
    backend = DenseAOResponseBackend(eri)
    other = UHFResponseOperator(
        UHFResponseOperator.build_problem(swapped, backend), backend
    )
    np.testing.assert_allclose(
        diagnose_uhf_stability(operator).eigenvalues,
        diagnose_uhf_stability(other).eigenvalues,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "options",
    [
        StabilityOptions(max_dimension=6),
        StabilityOptions(max_evaluations=14),
        StabilityOptions(budget_bytes=1),
    ],
)
def test_budget_precedes_any_action_validation_or_large_allocation(
    options: StabilityOptions, monkeypatch: pytest.MonkeyPatch
) -> None:
    operator, _ = _operator()
    operator.apply = Mock(side_effect=AssertionError("unexpected action"))
    operator.validate_current = Mock(
        side_effect=AssertionError("unexpected validation")
    )
    monkeypatch.setattr(
        np, "empty", Mock(side_effect=AssertionError("unexpected allocation"))
    )
    with pytest.raises(StabilityUnqualified, match="budget rejected"):
        diagnose_uhf_stability(operator, options=options)
    operator.apply.assert_not_called()
    operator.validate_current.assert_not_called()


@pytest.mark.parametrize(
    "failure", ["asymmetric", "nan", "complex", "shape", "late", "drift"]
)
def test_bad_actions_publish_no_certificate(failure: str) -> None:
    operator, _ = _operator()
    apply = operator.apply
    calls = 0

    def bad(vector: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        value = apply(vector).copy()
        if failure == "asymmetric":
            value[0] += vector[1]
        elif failure == "nan":
            value[0] = np.nan
        elif failure == "complex":
            value = value.astype(complex)
        elif failure == "shape":
            value = value[:-1]
        elif failure == "late" and calls > operator.dimension:
            raise RuntimeError("provider expired during residual")
        elif failure == "drift" and calls > operator.dimension:
            value *= 1.01
        return value

    operator.apply = bad
    with pytest.raises(StabilityUnqualified):
        diagnose_uhf_stability(operator)


def test_reference_hamiltonian_binding_and_late_stale() -> None:
    operator, _ = _operator()
    cert = diagnose_uhf_stability(operator)
    operator.backend.identity = "changed-Hamiltonian"
    with pytest.raises(StabilityUnqualified, match="stale"):
        cert.assert_current(operator)
    with pytest.raises(StabilityUnqualified):
        diagnose_uhf_stability(operator)
    operator, _ = _operator()
    operator.backend.eri = operator.backend.eri * 2
    with pytest.raises(StabilityUnqualified, match="reference Focks"):
        diagnose_uhf_stability(operator)


def test_threshold_controls_and_immutable_owned_direction() -> None:
    operator, _ = _operator(interacting=False)
    n = operator.dimension
    physical = np.eye(n)
    physical[0, 0] = -2e-6
    buffer = np.zeros(n)

    def apply(vector: np.ndarray) -> np.ndarray:
        buffer[:] = physical @ vector / 2
        return buffer

    operator.apply = apply
    negative = diagnose_uhf_stability(operator)
    assert negative.status == "UNSTABLE"
    assert negative.direction is not None
    buffer[:] = 42
    np.testing.assert_array_equal(negative.direction, np.eye(n)[0])
    for array in (negative.direction, negative.eigenvalues):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    assert (
        diagnose_uhf_stability(
            operator, options=StabilityOptions(curvature_tolerance=3e-6)
        ).status
        == "NEAR_SINGULAR"
    )
    physical[0, 0] = 5e-7
    assert diagnose_uhf_stability(operator).status == "NEAR_SINGULAR"
    physical[0, 0] = 2e-6
    assert diagnose_uhf_stability(operator).status == "STABLE"


def test_late_identity_change_and_empty_provider_failure() -> None:
    operator, _ = _operator()
    apply = operator.apply

    def changed(vector: np.ndarray) -> np.ndarray:
        result = apply(vector)
        operator.backend.identity = "expired"
        return result

    operator.apply = changed
    with pytest.raises(StabilityUnqualified):
        diagnose_uhf_stability(operator)
    empty, _ = _operator(4, 4)
    empty.validate_current = Mock(side_effect=RuntimeError("closed"))
    with pytest.raises(StabilityUnqualified, match="no partial"):
        diagnose_uhf_stability(empty)


def test_stricter_stationarity_and_uks_exclusion() -> None:
    operator, eri = _operator()
    r = replace(
        operator.problem.reference, scf_residual=5e-8, validation_tolerance=1e-7
    )
    backend = DenseAOResponseBackend(eri)
    other = UHFResponseOperator(UHFResponseOperator.build_problem(r, backend), backend)
    with pytest.raises(StabilityUnqualified, match="stationarity"):
        diagnose_uhf_stability(other)
    from tools.generativeqc_response.uhf import UKSResponseOperator

    with pytest.raises(StabilityUnqualified, match="shared canonical real UHF"):
        diagnose_uhf_stability(object.__new__(UKSResponseOperator))


def test_reference_error_prevents_overprecise_stable_claim() -> None:
    operator, eri = _operator(interacting=False)
    r = operator.problem.reference
    # Accepted tiny canonical error must enter the curvature uncertainty,
    # independently of an otherwise exact eigensystem.
    r = replace(r, orbital_energies_alpha=r.orbital_energies_alpha + 1e-9)
    backend = DenseAOResponseBackend(eri)
    operator = UHFResponseOperator(
        UHFResponseOperator.build_problem(r, backend), backend
    )
    operator.apply = lambda v: v * 1e-9
    result = diagnose_uhf_stability(
        operator, options=StabilityOptions(curvature_tolerance=1e-10)
    )
    assert result.reference_curvature_bound > 4e-9
    assert result.uncertainty >= result.reference_curvature_bound
    assert result.status == "NEAR_SINGULAR"


def _pyscf() -> tuple:
    try:
        import pyscf
        from pyscf import gto, scf
    except ImportError:
        if os.environ.get("GENERATIVEQC_REQUIRE_UHF_STABILITY_ORACLE") == "1":
            pytest.fail("pinned PySCF molecular gate required")
        pytest.skip("pinned PySCF molecular gate unavailable; not scientific PASS")
    assert pyscf.__version__ == "2.14.0"
    return gto, scf


@pytest.mark.parametrize("case", ["h2-saddle", "h2-broken", "li-doublet"])
def test_molecular_pinned_pyscf_energy_oracle(
    case: str, record_property: typing.Any
) -> None:
    gto, scf = _pyscf()
    mol = gto.M(
        atom="Li 0 0 0" if case == "li-doublet" else "H 0 0 0; H 0 0 2.0",
        basis="sto-3g",
        spin=1 if case == "li-doublet" else 0,
        verbose=0,
    )
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    if case == "h2-broken":
        # Atom-localized alpha/beta start, with no diagnostic-driven recovery.
        dm = np.zeros((2, mol.nao, mol.nao))
        dm[0, 0, 0], dm[1, 1, 1] = 1.0, 1.0
        mf.kernel(dm0=dm)
    elif case == "h2-saddle":
        rhf = scf.RHF(mol).run(conv_tol=1e-13)
        mf.kernel(dm0=np.stack([rhf.make_rdm1() / 2] * 2))
        assert np.linalg.norm(mf.make_rdm1()[0] - mf.make_rdm1()[1]) < 1e-10
    else:
        mf.kernel()
    assert mf.converged
    c, eps, occ = mf.mo_coeff, mf.mo_energy, mf.mo_occ
    fock = mf.get_fock(dm=mf.make_rdm1())
    overlap = mf.get_ovlp()
    residual = max(
        float(np.max(np.abs(f @ cs - overlap @ cs * es)))
        for f, cs, es in zip(fock, c, eps, strict=True)
    )
    assert residual < 1e-8
    r = UHFReferenceSnapshot(
        overlap,
        mf.get_hcore(),
        *fock,
        *c,
        *eps,
        *occ,
        mf.e_tot,
        residual,
        case,
        "sto-3g-pyscf-2.14.0",
        "converged-pyscf-oracle",
        representation="real_spherical",
    )
    eri = mol.intor("int2e", aosym="s1")
    backend = DenseAOResponseBackend(eri)
    operator = UHFResponseOperator(
        UHFResponseOperator.build_problem(r, backend), backend
    )
    result = diagnose_uhf_stability(operator)
    oracle = _energy_hessian(operator, eri)
    np.testing.assert_allclose(
        _rotated_energy(operator, eri, np.zeros(operator.dimension)) + mol.energy_nuc(),
        mf.e_tot,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.eigenvalues, np.linalg.eigvalsh(oracle), atol=3e-7, rtol=3e-6
    )
    # PySCF packs virtual-major occupied-minor. Check its *full* real internal
    # Hessian, 2*hop, with an explicit coordinate permutation. A single
    # symmetric Davidson seed at an exactly symmetric saddle can miss the
    # antisymmetric spin mode; disable seed symmetry in the extra status check.
    from pyscf.scf.stability import uhf_internal
    from pyscf.soscf.newton_ah import gen_g_hop_uhf

    _, hop, _ = gen_g_hop_uhf(mf, c, occ, with_symmetry=False)
    permutation = []
    offset = 0
    for spin in ("alpha", "beta"):
        no = r.nocc(spin)
        count = no * (r.nbf - no)
        permutation.extend(
            np.arange(offset, offset + count).reshape(no, r.nbf - no).T.ravel()
        )
        offset += count
    permutation = np.asarray(permutation)
    pyscf_matrix = np.column_stack([2 * hop(v) for v in np.eye(operator.dimension)])
    np.testing.assert_allclose(
        pyscf_matrix, oracle[np.ix_(permutation, permutation)], atol=3e-7, rtol=3e-6
    )
    _, internal_stable = uhf_internal(
        mf, with_symmetry=False, return_status=True, tol=1e-10
    )
    assert bool(internal_stable) == (case != "h2-saddle")
    assert result.status == ("UNSTABLE" if case == "h2-saddle" else "STABLE")
    if case == "h2-broken":
        symmetric_energy = scf.RHF(mol).run(conv_tol=1e-13).e_tot
        assert mf.e_tot < symmetric_energy - 0.1
    if result.direction is not None:
        v = result.direction
        np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-14)
        assert not v.flags.writeable
        with pytest.raises(ValueError):
            v.setflags(write=True)
        np.testing.assert_allclose(oracle @ v, result.eigenvalues[0] * v, atol=3e-7)
        origin = _rotated_energy(operator, eri, np.zeros(operator.dimension))
        curvatures = []
        for step in (1e-3, 2e-3):
            assert _rotated_energy(operator, eri, step * v) < origin - 1e-8
            curvature = (
                _rotated_energy(operator, eri, step * v)
                + _rotated_energy(operator, eri, -step * v)
                - 2 * origin
            ) / step**2
            curvatures.append(curvature)
        # Cancel the O(t^2) central-difference truncation rather than loosening
        # a physical-Hessian gate at the larger displacement.
        extrapolated = (4 * curvatures[0] - curvatures[1]) / 3
        np.testing.assert_allclose(extrapolated, result.eigenvalues[0], atol=1e-7)
    record_property("pyscf_version", "2.14.0")
    record_property("case", case)
    record_property("reference_energy", float(mf.e_tot))
    record_property("scf_residual", residual)
    record_property("status", result.status)
    record_property("minimum_curvature", float(result.eigenvalues[0]))
    record_property("certificate_identity", result.identity)
    record_property("dimension", result.dimension)
    record_property("response_actions", result.response_actions)
    record_property("reference_evaluations", result.reference_evaluations)
    record_property("backend_jk_calls", backend.statistics["actions"])
    assert backend.statistics["actions"] == 3 * (2 * operator.dimension + 1)
    record_property("accounted_bytes", result.accounted_bytes)
    record_property("endpoint_seconds", result.elapsed_seconds)
