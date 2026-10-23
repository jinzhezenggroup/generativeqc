"""GPU qualification of the guarded unset/default cold Direct J/K path."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="requires a finite Slurm allocation, RTX GPU and independent GPU4PySCF",
)


def test_default_cold_warm_reset_and_explicit_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No extra warm work, no implicit activation under explicit 0/off."""
    import cupy as cp
    from gpu4pyscf.dft import rks
    from pyscf import gto, lib

    from benchmarks.readme_defaults import native_calculator
    from benchmarks.readme_wb97mv import configure_reference_full_fock

    assert os.environ.get("SLURM_JOB_ID")
    assert "CUDA_VISIBLE_DEVICES" in os.environ
    water = (("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.8)), ("H", (1.7, 0.0, -0.6)))
    atoms = [
        (symbol, (xyz[0] + 6.0 * i, xyz[1], xyz[2]))
        for i in range(4)
        for symbol, xyz in water
    ]
    outcomes = {}
    for choice in ("unset", "off"):
        with monkeypatch.context() as scope:
            scope.delenv("GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK", raising=False)
            scope.delenv("GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK", raising=False)
            if choice == "off":
                scope.setenv("GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK", "off")
            calculator = native_calculator("pbe0", "standard", "direct", tmp_path)
            records = []
            try:
                with calculator.prepare_batch([atoms], warm_start=True) as owner:
                    for phase in ("cold", "warm", "reset"):
                        if phase == "reset":
                            owner.clear_warm_starts()
                        result = owner.execute(
                            strict=True, properties=("energy", "forces")
                        ).items[0]
                        assert result.converged and result.executed_backend == "cuda"
                        active = choice == "unset" and phase != "warm"
                        assert result.incremental_direct_jk["requested"] == (
                            choice == "unset"
                        )
                        assert result.incremental_direct_jk["active"] == active
                        assert result.warm_start_used == (phase == "warm")
                        records.append(
                            (float(result.energy), np.array(result.forces, copy=True))
                        )
            finally:
                calculator.clear_cache()
            outcomes[choice] = records
    lib.num_threads(1)
    mol = gto.M(atom=atoms, unit="Bohr", basis="def2-svp", cart=False, verbose=0)
    reference = rks.RKS(mol)
    reference.xc = "pbe0"
    reference.conv_tol = 1e-12
    reference.conv_tol_grad = 1e-9
    reference.direct_scf_tol = 1e-14
    reference.max_cycle = 150
    configure_reference_full_fock(reference)
    energy = float(reference.kernel())
    assert reference.converged
    gradient = reference.nuc_grad_method()
    gradient.grid_response = True
    forces = -cp.asnumpy(gradient.kernel())
    for choice, records in outcomes.items():
        assert len(records) == 3, choice
        for actual_energy, actual_forces in records:
            delta = actual_forces - forces
            assert abs(actual_energy - energy) / len(atoms) < 2.5e-6
            assert np.max(np.abs(delta)) < 2.5e-5
            assert np.sqrt(np.mean(delta**2)) < 7.5e-6
