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


@pytest.mark.parametrize("exchange", (False, True))
def test_provisional_screening_preserves_prepared_operator(
    exchange: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Signed delta actions cannot alter subsequent physical J/K on the owner."""
    from dataclasses import replace

    from generativeqc.fock import FockBuildSpec, FockTerm
    from generativeqc_compiler.dft import NativeAO
    from md_j_resident_probe import ResidentMdJProbe

    from benchmarks._md_j_reference import reference_molecule

    assert os.environ.get("SLURM_JOB_ID")
    monkeypatch.delenv("GENERATIVEQC_DISABLE_MD_J", raising=False)
    atoms = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.8)), ("H", (1.7, 0.0, -0.6))]
    spec = FockBuildSpec.hf("restricted", derivative_order=0)
    if not exchange:
        spec = replace(spec, exchange=FockTerm(False))
    with NativeAO(atoms, "def2-svp", representation="spherical") as basis:
        mol, scales = reference_molecule(basis)
        eri = mol.intor("int2e") * np.einsum(
            "i,j,k,l->ijkl", scales, scales, scales, scales
        )
        rng = np.random.default_rng(217)
        density = rng.normal(size=(basis.nao, basis.nao)) * 0.03
        density += density.T
        expected_j = np.einsum("ijkl,kl->ij", eri, density)
        expected_k = np.einsum("ikjl,kl->ij", eri, density)
        with ResidentMdJProbe(basis, spec, screening_tolerance=1e-13) as owner:
            original = owner.evaluate(density)
            for scale in (1e-3, -2e-4):
                proposed = owner.evaluate_proposal(scale * density, 1e-9)
                np.testing.assert_allclose(
                    proposed.coulomb, scale * expected_j, atol=2e-7, rtol=0
                )
                if exchange:
                    np.testing.assert_allclose(
                        proposed.exchange, scale * expected_k, atol=2e-7, rtol=0
                    )
            for invalid in (-1.0, 1e-14, float("nan"), float("inf")):
                with pytest.raises(RuntimeError, match="screening"):
                    owner.evaluate_proposal(density, invalid)
            restored = owner.evaluate(density)
            np.testing.assert_allclose(restored.coulomb, expected_j, atol=3e-11, rtol=0)
            np.testing.assert_allclose(
                restored.coulomb, original.coulomb, atol=3e-13, rtol=0
            )
            if exchange:
                np.testing.assert_allclose(
                    restored.exchange, expected_k, atol=3e-11, rtol=0
                )
                np.testing.assert_allclose(
                    restored.exchange, original.exchange, atol=3e-13, rtol=0
                )
        # Refusal leaves the ordinary producer usable when MD-J is unavailable.
        monkeypatch.setenv("GENERATIVEQC_DISABLE_MD_J", "1")
        with ResidentMdJProbe(basis, spec, screening_tolerance=1e-13) as owner:
            with pytest.raises(RuntimeError, match="MD-J"):
                owner.evaluate_proposal(density, 1e-9)
            np.testing.assert_allclose(
                owner.evaluate(density).coulomb, expected_j, atol=3e-11, rtol=0
            )
