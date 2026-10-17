"""CPU saddle recovery with an independent, analytically soluble Hubbard dimer.

The two-site model has a real UHF broken-symmetry singlet when U > 2*t.
The test SCF solver is separate from the production recovery controller.
"""

import typing
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from tools.generativeqc_response import (
    DenseAOResponseBackend,
    UHFReferenceSnapshot,
    UHFResponseOperator,
)
from tools.generativeqc_response.recovery import (
    RecoveryOptions,
    RecoveryUnqualified,
    recover_uhf_stability,
)
from tools.generativeqc_response.stability import (
    StabilityOptions,
    diagnose_uhf_stability,
)

T, U = 0.2, 1.2
HCORE = np.array([[0.0, -T], [-T, 0.0]])
ERI = np.zeros((2, 2, 2, 2))
ERI[0, 0, 0, 0] = ERI[1, 1, 1, 1] = U


def _focks(density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return HCORE + np.diag(U * np.diag(density[1])), HCORE + np.diag(
        U * np.diag(density[0])
    )


def _energy(density: np.ndarray) -> float:
    return float(
        np.sum(HCORE * (density[0] + density[1]))
        + U
        * (density[0, 0, 0] * density[1, 0, 0] + density[0, 1, 1] * density[1, 1, 1])
    )


def _canonical_operator(density: np.ndarray, *, generation: str) -> UHFResponseOperator:
    focks = _focks(density)
    frames = [np.linalg.eigh(f) for f in focks]
    orbitals = np.stack([cs[:, :1] @ cs[:, :1].T for _, cs in frames])
    # The reference must be a stationary fixed point, not just a diagonal Fock.
    np.testing.assert_allclose(orbitals, density, rtol=0, atol=3e-9)
    reference = UHFReferenceSnapshot(
        np.eye(2),
        HCORE,
        *focks,
        *(cs for _, cs in frames),
        *(values for values, _ in frames),
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
        _energy(orbitals),
        float(np.max(np.abs(orbitals - density))),
        "hubbard-dimer-t02-u12",
        "orthonormal-sites",
        generation,
    )
    backend = DenseAOResponseBackend(ERI)
    return UHFResponseOperator(
        UHFResponseOperator.build_problem(reference, backend), backend
    )


def _symmetric() -> UHFResponseOperator:
    _, c = np.linalg.eigh(HCORE)
    density = c[:, :1] @ c[:, :1].T
    return _canonical_operator(
        np.stack((density, density)), generation="symmetric-saddle"
    )


def _solve(seed: np.ndarray) -> UHFResponseOperator:
    # Independent fixed-point UHF SCF with damping; no PySCF or production
    # recovery code executes this test-only SCF.
    density = np.array(seed, dtype=np.float64)
    for _ in range(800):
        frames = [np.linalg.eigh(f)[1][:, :1] for f in _focks(density)]
        target = np.stack([c @ c.T for c in frames])
        if np.max(np.abs(target - density)) < 8e-12:
            return _canonical_operator(target, generation="reconverged-dimer")
        density = 0.55 * density + 0.45 * target
    raise RuntimeError("independent dimer SCF did not converge")


def test_negative_mode_recovers_lower_stationary_root_with_analytic_energy() -> None:
    origin = _symmetric()
    assert diagnose_uhf_stability(origin).status == "UNSTABLE"
    result = recover_uhf_stability(origin, _solve)
    assert result.status == "RECOVERED"
    assert result.recovered.status == "STABLE"
    assert result.energy_drop > 0.25
    assert result.attempted_seeds >= 1
    assert result.distinct_roots == 1
    assert result.equivalent_restarts >= 1  # alpha/beta inverted localization
    assert result.spin.spin_projection == 0
    assert 0.7 < result.spin.spin_squared < 1.0
    np.testing.assert_allclose(
        result.operator.problem.reference.reference_energy, -2 * T**2 / U, atol=2e-9
    )
    assert (
        np.linalg.norm(result.spin.alpha_ao_population - result.spin.beta_ao_population)
        > 0.5
    )
    result.assert_current()


def test_stable_root_is_unchanged_and_does_not_call_solver() -> None:
    broken = _solve(np.array([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]))
    assert diagnose_uhf_stability(broken).status == "STABLE"
    solver = Mock(side_effect=AssertionError("solver must not run"))
    result = recover_uhf_stability(broken, solver)
    assert result.status == "UNCHANGED" and result.energy_drop == 0
    assert result.attempted_seeds == 0
    result.assert_current()
    solver.assert_not_called()


def test_callback_must_reconverge_to_a_lower_stable_root() -> None:
    origin = _symmetric()
    with pytest.raises(RecoveryUnqualified, match="no lower-energy"):
        recover_uhf_stability(origin, lambda _: origin)
    with pytest.raises(RecoveryUnqualified, match="callback did not return"):
        recover_uhf_stability(origin, lambda _: None)

    def changed_source(seed: np.ndarray) -> UHFResponseOperator:
        candidate = _solve(seed)
        candidate.backend.identity = "wrong-hamiltonian"
        return candidate

    with pytest.raises(RecoveryUnqualified, match="Hamiltonian"):
        recover_uhf_stability(origin, changed_source)


def test_source_lifecycle_and_reported_energy_are_fail_closed() -> None:
    origin = _symmetric()

    def wrong_energy(seed: np.ndarray) -> UHFResponseOperator:
        candidate = _solve(seed)
        reference = replace(candidate.problem.reference, reference_energy=99.0)
        return UHFResponseOperator(
            UHFResponseOperator.build_problem(reference, candidate.backend),
            candidate.backend,
        )

    with pytest.raises(RecoveryUnqualified, match="SCF energy"):
        recover_uhf_stability(origin, wrong_energy)

    def expires_provider(seed: np.ndarray) -> UHFResponseOperator:
        origin.backend.identity = "changed-during-SCF"
        raise RuntimeError("SCF failed")

    with pytest.raises(RecoveryUnqualified):
        recover_uhf_stability(origin, expires_provider)


@pytest.mark.parametrize(
    "options",
    [
        RecoveryOptions(max_nbf=1),
        RecoveryOptions(budget_bytes=16),
        RecoveryOptions(budget_bytes=1800),
        RecoveryOptions(budget_bytes=6528),
        RecoveryOptions(budget_bytes=7000),
    ],
)
def test_budget_rejects_before_diagnostic_and_callback(
    options: RecoveryOptions,
) -> None:
    origin = _symmetric()
    origin.apply = Mock(side_effect=AssertionError("response action forbidden"))
    origin.backend.coulomb_exchange = Mock(
        side_effect=AssertionError("source evaluation forbidden")
    )
    solver = Mock(side_effect=AssertionError("solver forbidden"))
    with pytest.raises(RecoveryUnqualified, match="budget rejected"):
        recover_uhf_stability(origin, solver, options=options)
    origin.apply.assert_not_called()
    origin.backend.coulomb_exchange.assert_not_called()
    solver.assert_not_called()


def test_stricter_stability_budget_is_preserved() -> None:
    origin = _symmetric()
    origin.backend.coulomb_exchange = Mock(
        side_effect=AssertionError("source evaluation forbidden")
    )
    with pytest.raises(RecoveryUnqualified, match="budget rejected"):
        recover_uhf_stability(
            origin, _solve, stability_options=StabilityOptions(budget_bytes=1)
        )
    origin.backend.coulomb_exchange.assert_not_called()


def test_diagnostic_workspace_is_part_of_recovery_budget() -> None:
    origin = _symmetric()
    result = recover_uhf_stability(
        origin, _solve, options=RecoveryOptions(budget_bytes=12288)
    )
    assert result.initial.accounted_bytes <= result.initial.options.budget_bytes
    assert result.recovered.accounted_bytes <= result.recovered.options.budget_bytes
    assert result.initial.options.budget_bytes < 12288
    result.assert_current()


@pytest.mark.parametrize("target", ["candidate", "original", "retained"])
def test_provider_growth_exceeds_reserved_budget(target: str) -> None:
    origin = _symmetric()
    candidates = []

    def larger_provider(seed: np.ndarray) -> UHFResponseOperator:
        candidate = _solve(seed)
        if target == "candidate":
            candidate.backend.host_workspace_bytes += 1
        elif target == "original":
            origin.backend.host_workspace_bytes += 1
        elif candidates:
            candidates[0].backend.host_workspace_bytes += 1
        candidates.append(candidate)
        return candidate

    with pytest.raises(RecoveryUnqualified, match="provider budget"):
        recover_uhf_stability(origin, larger_provider)


@pytest.mark.parametrize("recover", [False, True])
def test_closed_native_owner_invalidates_recovery_evidence(recover: bool) -> None:
    """Exercise native-owner lifetime without loading a native library."""
    model = (
        _symmetric()
        if recover
        else _solve(np.array([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]))
    )
    operator, check_open = _native_owner_operator(model)

    def solve(seed: np.ndarray) -> UHFResponseOperator:
        candidate = _solve(seed)
        return UHFResponseOperator(
            UHFResponseOperator.build_problem(
                candidate.problem.reference, operator.backend
            ),
            operator.backend,
        )

    result = recover_uhf_stability(operator, solve if recover else Mock())
    assert result.status == ("RECOVERED" if recover else "UNCHANGED")
    result.assert_current()
    check_open.side_effect = RuntimeError("source closed")
    with pytest.raises(RecoveryUnqualified, match="source"):
        result.assert_current()


def _native_owner_operator(
    model: UHFResponseOperator,
) -> tuple[UHFResponseOperator, Mock]:
    from types import SimpleNamespace

    from tools.generativeqc_response.backends import NativeJKBackend

    reference = model.problem.reference
    check_open = Mock()
    source = SimpleNamespace(
        nbf=reference.nbf,
        identity="finite-dimer-source-owner",
        geometry_hash=reference.geometry_hash,
        basis_hash=reference.basis_hash,
        representation=reference.representation,
        _check_open=check_open,
    )
    backend = NativeJKBackend(source)
    # Only the integral action is supplied by the independent finite model;
    # metadata validation and source ownership use the real native backend.
    backend.coulomb_exchange = model.backend.coulomb_exchange
    operator = UHFResponseOperator(
        UHFResponseOperator.build_problem(reference, backend), backend
    )
    return operator, check_open


def test_source_closed_after_one_root_prevents_partial_success() -> None:
    origin, check_open = _native_owner_operator(_symmetric())
    calls = 0

    def closes_after_root(seed: np.ndarray) -> UHFResponseOperator:
        nonlocal calls
        calls += 1
        if calls == 2:
            check_open.side_effect = RuntimeError("source closed")
            raise RuntimeError("SCF failed after closing its source")
        candidate = _solve(seed)
        return UHFResponseOperator(
            UHFResponseOperator.build_problem(
                candidate.problem.reference, origin.backend
            ),
            origin.backend,
        )

    with pytest.raises(RecoveryUnqualified, match="source"):
        recover_uhf_stability(origin, closes_after_root)
    assert calls == 2


@pytest.mark.parametrize(
    "angles",
    [
        (),
        (-0.1,),
        (0.0,),
        (1.6,),
        (0.3, 0.2),
        (float("nan"),),
        (0.1,) * 7,
    ],
)
def test_invalid_angle_admission(angles: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        RecoveryOptions(angles=angles)


def test_native_h2_broken_root_matches_independent_pyscf_214(
    record_property: typing.Any,
) -> None:
    """Real native SCF reconvergence; PySCF is used only as a test oracle."""
    import os

    try:
        import pyscf
        from pyscf import gto, scf
    except ImportError:
        if os.environ.get("GENERATIVEQC_REQUIRE_UHF_RECOVERY_ORACLE") == "1":
            pytest.fail("pinned PySCF recovery oracle required")
        pytest.skip("pinned molecular oracle unavailable; not scientific PASS")
    assert pyscf.__version__ == "2.14.0"

    from generativeqc.fock import FockBuildSpec, FockPlan
    from generativeqc_compiler.dft import NativeAO
    from scipy.linalg import eigh

    from tools.generativeqc_posthf.sources import NativeSource
    from tools.generativeqc_response.backends import NativeJKBackend
    from tools.generativeqc_response.recovery import recover_uhf_with_native_fock_plan

    bohr = 1.8897261246257702
    atoms = [("H", (0.0, 0.0, -bohr)), ("H", (0.0, 0.0, bohr))]
    with (
        NativeAO(atoms, basis="sto-3g", multiplicity=1) as basis,
        NativeSource(atoms, basis="sto-3g", multiplicity=1) as source,
        FockPlan(
            basis,
            FockBuildSpec.hf("unrestricted", derivative_order=0),
            screening_tolerance=0.0,
            device="cpu",
        ) as native_plan,
    ):
        overlap, hcore = source.one_electron()
        rhf_density, _ = source.rhf_density(tolerance=1e-12)
        symmetric_seed = np.stack([0.5 * rhf_density] * 2)
        symmetric = native_plan.solve(
            initial_density=symmetric_seed,
            compute_forces=False,
            max_iterations=200,
            energy_tolerance=1e-11,
            density_tolerance=1e-10,
        )
        assert symmetric.initial_density_used
        assert np.max(np.abs(symmetric.density[0] - symmetric.density[1])) < 1e-8
        backend = NativeJKBackend(source)
        # This initial setup is test-only; production recovery uses the shared
        # UHF response Fock seam and an exact native FockPlan SCF callback.
        total_j, _ = backend.coulomb_exchange(symmetric.density.sum(axis=0))
        _, ka = backend.coulomb_exchange(symmetric.density[0])
        _, kb = backend.coulomb_exchange(symmetric.density[1])
        focks = (hcore + total_j - ka, hcore + total_j - kb)
        eigenpairs = [eigh(f, overlap) for f in focks]
        residual = max(
            float(np.max(np.abs(f @ c - overlap @ c * e)))
            for f, (e, c) in zip(focks, eigenpairs, strict=True)
        )
        snapshot = UHFReferenceSnapshot(
            overlap,
            hcore,
            *focks,
            *(c for e, c in eigenpairs),
            *(e for e, c in eigenpairs),
            *(np.array([1.0, 0.0]) for _ in (0, 1)),
            symmetric.energy,
            max(residual, symmetric.density_rms),
            source.geometry_hash,
            source.basis_hash,
            "symmetric-native-scaffold",
            representation="cartesian",
            hf_backend="cpu-native",
        )
        operator = UHFResponseOperator(
            UHFResponseOperator.build_problem(snapshot, backend), backend
        )
        assert diagnose_uhf_stability(operator).status == "UNSTABLE"
        result = recover_uhf_with_native_fock_plan(operator, native_plan)
        result.assert_current()
        assert result.recovered.status == "STABLE"
        assert result.energy_drop > 0.1
        assert result.spin.spin_projection == 0
        assert result.spin.spin_squared > 0.5
        # Basis exponents and contractions are aligned with the native source.
        # PySCF neither supplies the integrals nor performs production recovery.
        matched = {
            "H": [
                [
                    shell.angular_momentum,
                    *[[p.exponent, p.coefficient] for p in shell.primitives],
                ]
                for shell in basis.shells
                if shell.atom_index == 0
            ]
        }
        mol = gto.M(
            atom=atoms, unit="Bohr", basis=matched, spin=0, cart=True, verbose=0
        )
        oracle = scf.UHF(mol)
        oracle.conv_tol = 1e-13
        oracle.conv_tol_grad = 1e-10
        oracle.max_cycle = 200
        local_start = np.zeros((2, 2, 2))
        local_start[0, 0, 0] = local_start[1, 1, 1] = 1.0
        oracle.kernel(dm0=local_start)
        assert oracle.converged
        np.testing.assert_allclose(
            result.operator.problem.reference.reference_energy, oracle.e_tot, atol=2e-7
        )
        record_property(
            "native_recovered_energy",
            float(result.operator.problem.reference.reference_energy),
        )
        record_property("pyscf_version", pyscf.__version__)
        record_property("recovered_spin_squared", result.spin.spin_squared)
        record_property("source_evidence", result.identity)


@pytest.mark.parametrize(
    "transform",
    (
        np.eye(2),
        np.diag([1e3, 1e-3]),
        np.array([[12.0, 3.0], [0.0, 0.02]]),
        np.array([[0.0, -0.02], [12.0, 3.0]]),
    ),
)
@pytest.mark.parametrize("angle", (0.0, 1e-6, 1e-3))
@pytest.mark.parametrize("flip", (False, True))
def test_root_distance_matches_independent_physical_projectors(
    transform: np.ndarray, angle: float, flip: bool
) -> None:
    from tools.generativeqc_response.recovery import _same_root

    # Independent physical orthonormal frame; do not derive the expected
    # classification from the production overlap factorization.
    left = np.stack((np.diag([1.0, 0.0]), np.diag([0.0, 1.0])))
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    right = np.stack([rotation @ block @ rotation.T for block in left])
    expected = max(
        np.linalg.norm(a - b, "fro") for a, b in zip(left, right, strict=True)
    )
    if flip:
        right = right[::-1]
    inverse = np.linalg.inv(transform)
    ao_left = np.stack([inverse @ block @ inverse.T for block in left])
    ao_right = np.stack([inverse @ block @ inverse.T for block in right])
    factor = np.linalg.cholesky(transform.T @ transform)
    assert _same_root(ao_left, ao_right, factor, 1e-5, spin_flip=flip) == (
        expected < 1e-5
    )
    if flip:
        assert not _same_root(ao_left, ao_right, factor, 1e-5, spin_flip=False)


def _change_ao_basis(
    operator: UHFResponseOperator, transform: np.ndarray
) -> UHFResponseOperator:
    """Independent tensor change of coordinates, not a recovery implementation."""
    reference = operator.problem.reference
    inverse = np.linalg.inv(transform)
    updated = replace(
        reference,
        overlap=transform.T @ reference.overlap @ transform,
        hcore=transform.T @ reference.hcore @ transform,
        fock_alpha=transform.T @ reference.fock_alpha @ transform,
        fock_beta=transform.T @ reference.fock_beta @ transform,
        coefficients_alpha=inverse @ reference.coefficients_alpha,
        coefficients_beta=inverse @ reference.coefficients_beta,
        basis_hash="changed-coordinate-sites",
    )
    transformed_eri = np.einsum(
        "pi,qj,rk,sl,pqrs->ijkl", transform, transform, transform, transform, ERI
    )
    backend = DenseAOResponseBackend(transformed_eri)
    return UHFResponseOperator(
        UHFResponseOperator.build_problem(updated, backend), backend
    )


@pytest.mark.parametrize(
    "transform",
    (np.diag([2.0, 0.5]), np.array([[1.2, 0.3], [0.0, 0.8]])),
)
def test_recovery_and_spin_norm_are_covariant_under_ao_basis_changes(
    transform: np.ndarray,
) -> None:
    baseline = recover_uhf_stability(_symmetric(), _solve)

    def solve(seed: np.ndarray) -> UHFResponseOperator:
        physical = np.stack([transform @ block @ transform.T for block in seed])
        return _change_ao_basis(_solve(physical), transform)

    result = recover_uhf_stability(_change_ao_basis(_symmetric(), transform), solve)
    assert result.status == "RECOVERED"
    assert result.distinct_roots == baseline.distinct_roots == 1
    assert result.equivalent_restarts == baseline.equivalent_restarts
    np.testing.assert_allclose(result.energy_drop, baseline.energy_drop, atol=1e-10)
    np.testing.assert_allclose(
        result.spin.spin_squared, baseline.spin.spin_squared, atol=1e-10
    )
    np.testing.assert_allclose(
        result.spin.spin_density_norm, baseline.spin.spin_density_norm, atol=1e-10
    )
    # The analytic dimer has <S^2> = 1 - (2t/U)^2 and a singlet spin-density
    # projector norm squared of 2<S^2>. This oracle does not use AO whitening.
    expected_s2 = 1 - (2 * T / U) ** 2
    np.testing.assert_allclose(
        result.spin.spin_density_norm, np.sqrt(2 * expected_s2), atol=1e-9
    )
    for spin, population in (
        ("alpha", result.spin.alpha_ao_population),
        ("beta", result.spin.beta_ao_population),
    ):
        np.testing.assert_allclose(
            np.sum(population), result.operator.problem.reference.nocc(spin), atol=1e-10
        )
    result.assert_current()
    # Stable/no-change publication must use the identical diagnostic metric.
    stable = recover_uhf_stability(
        result.operator, Mock(side_effect=AssertionError("unexpected restart"))
    )
    assert stable.status == "UNCHANGED"
    np.testing.assert_allclose(
        stable.spin.spin_density_norm, result.spin.spin_density_norm, atol=1e-12
    )
    stable.assert_current()


def test_molecular_root_metric_matches_pinned_pyscf_occupied_subspaces() -> None:
    import os

    try:
        import pyscf
        from pyscf import gto, lib, scf
    except ImportError:
        if os.environ.get("GENERATIVEQC_REQUIRE_UHF_RECOVERY_ORACLE") == "1":
            pytest.fail("pinned molecular root-metric oracle required")
        pytest.skip("pinned molecular root-metric oracle unavailable")
    assert pyscf.__version__ == "2.14.0"
    from tools.generativeqc_response.recovery import _metrics, _same_root

    lib.num_threads(1)
    molecule = gto.M(
        atom="H 0 0 0; H 0 0 2", basis="sto-3g", unit="Angstrom", spin=0, verbose=0
    )
    mf = scf.UHF(molecule)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-11
    mf.kernel(dm0=np.array([np.diag([1.0, 0.0]), np.diag([0.0, 1.0])]))
    assert mf.converged
    overlap = mf.get_ovlp()
    coefficients = mf.mo_coeff
    densities = mf.make_rdm1()
    expected_s2 = mf.spin_square()[0]
    for transform in (
        np.eye(2),
        np.diag([1e3, 1e-3]),
        np.array([[12.0, 3.0], [0.0, 0.02]]),
    ):
        inverse = np.linalg.inv(transform)
        factor = np.linalg.cholesky(transform.T @ overlap @ transform)
        changed = np.stack([inverse @ d @ inverse.T for d in densities])
        # A normal immutable response reference carries all required metadata;
        # only its geometry-independent spin diagnostic is under test here.
        reference = replace(
            _symmetric().problem.reference,
            overlap=transform.T @ overlap @ transform,
            hcore=transform.T @ mf.get_hcore() @ transform,
            fock_alpha=transform.T @ mf.get_fock()[0] @ transform,
            fock_beta=transform.T @ mf.get_fock()[1] @ transform,
            coefficients_alpha=inverse @ coefficients[0],
            coefficients_beta=inverse @ coefficients[1],
            orbital_energies_alpha=mf.mo_energy[0],
            orbital_energies_beta=mf.mo_energy[1],
            reference_energy=mf.e_tot,
        )
        spin = _metrics(reference, changed, factor)
        np.testing.assert_allclose(spin.spin_squared, expected_s2, atol=1e-10)
        np.testing.assert_allclose(
            spin.spin_density_norm, np.sqrt(2 * expected_s2), atol=1e-10
        )
        angle = 1e-6
        rotated = [
            c[:, 0] * np.cos(angle) + c[:, 1] * np.sin(angle) for c in coefficients
        ]
        nearby = np.stack([inverse @ np.outer(c, c) @ inverse.T for c in rotated])
        assert _same_root(changed, nearby, factor, 1e-5, spin_flip=False)


def test_metric_failure_blocks_root_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.generativeqc_response.recovery import _metric_density_norm

    with pytest.raises(RecoveryUnqualified, match="nonfinite overlap-metric"):
        _metric_density_norm(np.full((2, 2), np.inf), np.eye(2))
    model = _symmetric()
    monkeypatch.setattr(
        np.linalg, "cholesky", Mock(side_effect=np.linalg.LinAlgError("singular"))
    )
    with pytest.raises(RecoveryUnqualified, match="metric factorization"):
        recover_uhf_stability(model, _solve)


@pytest.mark.parametrize("failure", ("hamiltonian", "response"))
def test_root_metric_does_not_relax_hamiltonian_or_response_qualification(
    failure: str,
) -> None:
    model = _change_ao_basis(_symmetric(), np.diag([2.0, 0.5]))
    if failure == "hamiltonian":
        original = model.backend.coulomb_exchange

        def mismatched(density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            coulomb, exchange = original(density)
            return coulomb + 1e-3 * np.eye(2), exchange

        model.backend.coulomb_exchange = mismatched
    else:
        original_apply = model.apply

        def inconsistent(vector: np.ndarray) -> np.ndarray:
            result = original_apply(vector)
            # A linear source must agree on non-basis directions too.
            if np.count_nonzero(vector) > 1:
                result = result + 1e-3
            return result

        model.apply = inconsistent
    solver = Mock(side_effect=AssertionError("SCF forbidden"))
    with pytest.raises(RecoveryUnqualified):
        recover_uhf_stability(model, solver)
    solver.assert_not_called()
