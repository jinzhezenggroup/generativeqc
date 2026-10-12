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
