"""Real-device executable-plan acceptance independent of scientific warm state."""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pytest
from generativeqc import Calculator

from tools.cc_gradient_fixtures import inputs
from tools.generate_validation_references import pyscf_molecule
from tools.validate_ccsd_t_gradient import _solve

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_CORRELATED_PLAN_CUDA_TEST") != "1",
    reason="requires an explicitly allocated CUDA device and native library",
)


def _geometry(changed: bool) -> dict:
    """Use the pinned explicit basis and distort one bond, not the whole frame."""
    value = inputs("h2")
    if changed:
        value["coordinates"][1][2] += 0.01
    return value


@lru_cache(maxsize=2)
def _coupled_reference(changed: bool) -> dict:
    """Reuse the corrected, converged independent triples-Lambda oracle."""
    pytest.importorskip("pyscf")
    pytest.importorskip("threadpoolctl")
    return _solve(_geometry(changed), gradients=True)


@lru_cache(maxsize=6)
def _reference(method: str, changed: bool) -> tuple[float, np.ndarray]:
    """Return independent complete energies and analytic forces at each frame."""
    if method != "mp2":
        reference = _coupled_reference(changed)
        if method == "rccsd":
            return (
                reference["total_energy"] - reference["triples_energy"],
                -np.asarray(reference["ccsd_gradient"]),
            )
        return reference["total_energy"], -np.asarray(reference["gradient"])

    pyscf = pytest.importorskip("pyscf")
    from pyscf import mp, scf

    assert pyscf.__version__ == "2.14.0"
    molecule, _, _ = pyscf_molecule(_geometry(changed))
    hartree_fock = scf.RHF(molecule)
    hartree_fock.conv_tol = 1e-13
    hartree_fock.conv_tol_grad = 1e-11
    hartree_fock.max_cycle = 200
    hartree_fock.kernel()
    assert hartree_fock.converged
    correlated = mp.MP2(hartree_fock).run()
    return correlated.e_tot, -np.asarray(correlated.nuc_grad_method().kernel())


@pytest.mark.parametrize("method", ("mp2", "rccsd", "ccsd(t)"))
@pytest.mark.parametrize("warm_start", (False, True))
def test_plan_replay_distorted_geometry_and_recovery_match_independent_forces(
    method: str, warm_start: bool
) -> None:
    """Every replay reconverges RHF while retaining compatible executable state.

    Disabling density reuse must not disable the executable owner. Full analytic
    E/F references at both geometries guard against stale-reference or stale-force
    publication, and failed input must not replace the last accepted warm state.
    """
    original = _geometry(False)
    moved = _geometry(True)
    atoms = list(zip(original["atomic_numbers"], original["coordinates"], strict=True))
    budget = 8 << 30
    calculator = Calculator(
        method=method,
        basis="sto-3g",
        basis_representation=original["basis_representation"],
        device="cuda",
        correlation_memory_budget_bytes=budget,
        max_iterations=200,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        ccsd_max_iterations=150,
        ccsd_energy_tolerance=1e-13,
        ccsd_residual_tolerance=1e-11,
    )
    with calculator.prepare_batch([atoms], warm_start=warm_start) as batch:
        for phase, changed, coordinates in (
            ("cold", False, None),
            ("stationary", False, None),
            ("changed", True, [moved["coordinates"]]),
            ("changed_warm", True, [moved["coordinates"]]),
        ):
            result = batch.execute(
                coordinates=coordinates, properties=("energy", "forces"), strict=True
            ).items[0]
            energy, forces = _reference(method, changed)
            assert result.succeeded and result.converged
            assert result.executed_backend == "cuda"
            assert result.warm_start_used == (warm_start and phase != "cold")
            assert not result.warm_start_fallback
            assert result.energy == pytest.approx(energy, abs=3e-9)
            np.testing.assert_allclose(result.forces, forces, atol=1e-6, rtol=0)
            diagnostic = result.correlation
            assert diagnostic is not None
            assert diagnostic.reference_execution_plan_reused == (phase != "cold")
            assert (
                0
                < diagnostic.reference_execution_plan_owned_device_bytes
                <= diagnostic.numeric_capacity_bytes
                <= budget
            )
            assert diagnostic.reference_residual <= 1e-9
            assert diagnostic.response_absolute_residual <= 1e-9
            if method != "mp2":
                assert diagnostic.ccsd_replay_singles_residual_max <= 1e-11
                assert diagnostic.ccsd_replay_doubles_residual_max <= 1e-11

        failed = batch.execute(
            coordinates=[np.zeros((1, 3))],
            properties=("energy", "forces"),
            strict=False,
        ).items[0]
        assert not failed.succeeded
        assert failed.correlation is None and failed.forces is None
        recovered = batch.execute(
            coordinates=[moved["coordinates"]],
            properties=("energy", "forces"),
            strict=True,
        ).items[0]
        energy, forces = _reference(method, True)
        assert recovered.warm_start_used == warm_start
        assert not recovered.warm_start_fallback
        assert recovered.correlation is not None
        assert recovered.correlation.reference_execution_plan_reused
        assert recovered.energy == pytest.approx(energy, abs=3e-9)
        np.testing.assert_allclose(recovered.forces, forces, atol=1e-6, rtol=0)
