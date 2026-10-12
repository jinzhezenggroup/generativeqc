"""CPU saddle recovery with an independent, analytically soluble Hubbard dimer.

The two-site model has a real UHF broken-symmetry singlet when U > 2*t.
The test SCF solver is separate from the production recovery controller.
"""

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
from tools.generativeqc_response.stability import diagnose_uhf_stability


T, U = 0.2, 1.2
HCORE = np.array([[0.0, -T], [-T, 0.0]])
ERI = np.zeros((2, 2, 2, 2))
ERI[0, 0, 0, 0] = ERI[1, 1, 1, 1] = U


def _focks(density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return HCORE + np.diag(U * np.diag(density[1])), HCORE + np.diag(U * np.diag(density[0]))


def _energy(density: np.ndarray) -> float:
    return float(np.sum(HCORE * (density[0] + density[1])) + U * (density[0, 0, 0] * density[1, 0, 0] + density[0, 1, 1] * density[1, 1, 1]))


def _canonical_operator(density: np.ndarray, *, generation: str) -> UHFResponseOperator:
    focks = _focks(density)
    frames = [np.linalg.eigh(f) for f in focks]
    orbitals = np.stack([cs[:, :1] @ cs[:, :1].T for _, cs in frames])
    # The reference must be a stationary fixed point, not just a diagonal Fock.
    np.testing.assert_allclose(orbitals, density, rtol=0, atol=3e-9)
    reference = UHFReferenceSnapshot(
        np.eye(2), HCORE, *focks, *(cs for _, cs in frames),
        *(values for values, _ in frames), np.array([1.0, 0.0]),
        np.array([1.0, 0.0]), _energy(orbitals),
        float(np.max(np.abs(orbitals - density))),
        "hubbard-dimer-t02-u12", "orthonormal-sites", generation,
    )
    backend = DenseAOResponseBackend(ERI)
    return UHFResponseOperator(UHFResponseOperator.build_problem(reference, backend), backend)


def _symmetric() -> UHFResponseOperator:
    _, c = np.linalg.eigh(HCORE)
    density = c[:, :1] @ c[:, :1].T
    return _canonical_operator(np.stack((density, density)), generation="symmetric-saddle")


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
    np.testing.assert_allclose(result.operator.problem.reference.reference_energy, -2 * T**2 / U, atol=2e-9)
    assert np.linalg.norm(result.spin.alpha_ao_population - result.spin.beta_ao_population) > 0.5
    result.assert_current()


def test_stable_root_is_unchanged_and_does_not_call_solver() -> None:
    broken = _solve(np.array([[[1., 0.], [0., 0.]], [[0., 0.], [0., 1.]]]))
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
        return UHFResponseOperator(UHFResponseOperator.build_problem(reference, candidate.backend), candidate.backend)

    with pytest.raises(RecoveryUnqualified, match="SCF energy"):
        recover_uhf_stability(origin, wrong_energy)

    def expires_provider(seed: np.ndarray) -> UHFResponseOperator:
        origin.backend.identity = "changed-during-SCF"
        raise RuntimeError("SCF failed")

    with pytest.raises(RecoveryUnqualified):
        recover_uhf_stability(origin, expires_provider)


@pytest.mark.parametrize("options", [
    RecoveryOptions(max_nbf=1), RecoveryOptions(budget_bytes=16),
])
def test_budget_rejects_before_diagnostic_and_callback(options: RecoveryOptions) -> None:
    origin = _symmetric()
    origin.apply = Mock(side_effect=AssertionError("response action forbidden"))
    solver = Mock(side_effect=AssertionError("solver forbidden"))
    with pytest.raises(RecoveryUnqualified, match="budget rejected"):
        recover_uhf_stability(origin, solver, options=options)
    origin.apply.assert_not_called()
    solver.assert_not_called()


@pytest.mark.parametrize("angles", [
    (), (-0.1,), (0.,), (1.6,), (0.3, 0.2), (float("nan"),), (0.1,) * 7,
])
def test_invalid_angle_admission(angles: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        RecoveryOptions(angles=angles)


def test_native_h2_broken_root_matches_independent_pyscf_214(record_property) -> None:
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

    from scipy.linalg import eigh

    from generativeqc.fock import FockBuildSpec, FockPlan
    from generativeqc_compiler.dft import NativeAO
    from tools.generativeqc_posthf.sources import NativeSource
    from tools.generativeqc_response.backends import NativeJKBackend
    from tools.generativeqc_response.recovery import recover_uhf_with_native_fock_plan

    bohr = 1.8897261246257702
    atoms = [("H", (0.0, 0.0, -bohr)), ("H", (0.0, 0.0, bohr))]
    with (
        NativeAO(atoms, basis="sto-3g", multiplicity=1) as basis,
        NativeSource(atoms, basis="sto-3g", multiplicity=1) as source,
        FockPlan(
            basis, FockBuildSpec.hf("unrestricted", derivative_order=0),
            screening_tolerance=0.0, device="cpu",
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
            overlap, hcore, *focks, *(c for e, c in eigenpairs),
            *(e for e, c in eigenpairs), *(np.array([1.0, 0.0]) for _ in (0, 1)),
            symmetric.energy, max(residual, symmetric.density_rms),
            source.geometry_hash, source.basis_hash, "symmetric-native-scaffold",
            representation="cartesian", hf_backend="cpu-native",
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
                [shell.angular_momentum, *[
                    [p.exponent, p.coefficient] for p in shell.primitives
                ]]
                for shell in basis.shells if shell.atom_index == 0
            ]
        }
        mol = gto.M(atom=atoms, unit="Bohr", basis=matched, spin=0, cart=True, verbose=0)
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
        record_property("native_recovered_energy", float(result.operator.problem.reference.reference_energy))
        record_property("pyscf_version", pyscf.__version__)
        record_property("recovered_spin_squared", result.spin.spin_squared)
        record_property("source_evidence", result.identity)
