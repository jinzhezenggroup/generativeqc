"""Qualify bounded incremental proposals and warm reuse against full-Fock SCF."""

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="requires a finite Slurm allocation and independent GPU4PySCF",
)


@pytest.mark.parametrize("method", ("pbe", "pbe0", "b3lyp", "r2scan", "m06-2x"))
def test_incremental_cold_warm_reset_against_full_density_reference(
    method: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only full physical closure may publish a proposal or reusable warm state.

    The native cold start is independent of the tighter oracle. Reusing and
    clearing that native state exercise per-solve policy restoration. Timing
    and exact iteration counts belong to the separate endpoint qualification.
    """
    import cupy as cp
    from gpu4pyscf.dft import rks
    from pyscf import gto, lib

    from benchmarks.readme_defaults import METHODS, native_calculator
    from benchmarks.readme_wb97mv import configure_reference_full_fock

    assert os.environ.get("SLURM_JOB_ID")
    assert "CUDA_VISIBLE_DEVICES" in os.environ
    monkeypatch.setenv("GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK", "1")
    monkeypatch.setenv("GENERATIVEQC_KS_INCREMENTAL_PROPOSALS", "1")
    # Zero disables requested periodic refresh, but screened proposal epochs
    # still have a finite implementation cap and mandatory full-density tail.
    monkeypatch.setenv("GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK_REBUILD_INTERVAL", "0")
    atoms = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 1.8)),
        ("H", (1.7, 0.0, -0.6)),
    ]
    calculator = native_calculator(method, "standard", "direct", tmp_path)
    samples = []
    try:
        with calculator.prepare_batch([atoms], warm_start=True) as owner:
            for phase in ("cold", "warm", "reset"):
                if phase == "reset":
                    owner.clear_warm_starts()
                item = owner.execute(
                    strict=True, properties=("energy", "forces")
                ).items[0]
                assert item.converged and item.executed_backend == "cuda"
                assert item.incremental_direct_jk["requested"]
                assert item.incremental_direct_jk["active"] == (phase != "warm")
                assert item.warm_start_used == (phase == "warm")
                assert not item.warm_start_fallback
                samples.append((float(item.energy), np.array(item.forces, copy=True)))
    finally:
        calculator.clear_cache()

    lib.num_threads(1)
    mol = gto.M(atom=atoms, unit="Bohr", basis="def2-svp", cart=False, verbose=0)
    reference = rks.RKS(mol)
    reference.xc = METHODS[method][1]
    reference.conv_tol = 1e-12
    reference.conv_tol_grad = 1e-9
    reference.direct_scf_tol = 1e-14
    reference.max_cycle = 150
    # RKS can reuse vj/vk tags even with direct_scf=False. Suppress both density
    # and potential history so every independent reference Fock is truly full.
    configure_reference_full_fock(reference)
    energy = float(reference.kernel())
    assert reference.converged
    gradient = reference.nuc_grad_method()
    gradient.grid_response = True
    forces = -cp.asnumpy(gradient.kernel())
    for actual_energy, actual_forces in samples:
        error = actual_forces - forces
        assert abs(actual_energy - energy) / len(atoms) < 2.5e-6
        assert np.max(np.abs(error)) < 2.5e-5
        assert np.sqrt(np.mean(error**2)) < 7.5e-6
