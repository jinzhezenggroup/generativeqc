"""GPU qualification of the guarded unset/default cold Direct J/K path."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="requires a finite Slurm allocation, RTX GPU and independent GPU4PySCF",
)


@pytest.mark.parametrize(
    "arm,mode",
    (("standard", "direct"), ("fast", "direct"), ("standard", "df"), ("fast", "df")),
)
def test_default_cold_warm_reset_and_explicit_off(
    arm: str, mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Qualify automatic admission and Fast/DF fallback at complete E/F endpoints.

    The DF oracle uses the same frozen auxiliary basis, rather than treating
    fitting error as solver error. Fast must retain its own precision receipt.
    A moved seed cannot borrow automatic cold admission or a frozen warm proof.
    """
    import cupy as cp
    from gpu4pyscf.dft import rks
    from pyscf import gto, lib

    from benchmarks.readme_defaults import native_calculator, prepare_inputs
    from benchmarks.readme_hf_scaling import scaling_cases
    from benchmarks.readme_wb97mv import configure_reference_full_fock

    assert os.environ.get("SLURM_JOB_ID")
    assert "CUDA_VISIBLE_DEVICES" in os.environ
    atoms = scaling_cases()["water-12"].atoms
    coordinates = np.array([xyz for _, xyz in atoms])
    moved = coordinates.copy()
    moved[1, 0] += 0.02
    if mode == "df":
        prepare_inputs(tmp_path)
    outcomes = {}
    for choice in ("unset", "off"):
        with monkeypatch.context() as scope:
            scope.delenv("GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK", raising=False)
            scope.delenv("GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK", raising=False)
            if choice == "off":
                scope.setenv("GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK", "off")
            calculator = native_calculator("pbe0", arm, mode, tmp_path)
            records = []
            try:
                with calculator.prepare_batch([atoms], warm_start=True) as owner:
                    for phase in ("cold", "warm", "moved", "moved-warm", "reset"):
                        if phase == "reset":
                            owner.clear_warm_starts()
                        if phase == "moved":
                            owner.set_warm_start_updates(True)
                        # None selects the originally prepared coordinates;
                        # every replay of the displaced geometry must name it.
                        result = owner.execute(
                            [moved]
                            if phase in ("moved", "moved-warm", "reset")
                            else None,
                            strict=True,
                            properties=("energy", "forces"),
                        ).items[0]
                        assert result.converged and result.executed_backend == "cuda"
                        requested = (
                            choice == "unset" and arm == "standard" and mode == "direct"
                        )
                        active = requested and phase in ("cold", "reset")
                        assert result.incremental_direct_jk["requested"] == requested
                        assert result.incremental_direct_jk["active"] == active
                        if phase in ("warm", "moved-warm"):
                            assert result.warm_start_used
                            assert result.incremental_direct_jk["delta_builds"] == 0
                        if active:
                            assert result.incremental_direct_jk["delta_builds"] > 0
                            work = result.incremental_direct_jk
                            assert work["post_scf_full_builds"] > 0
                            assert (
                                work["anchor_full_builds"]
                                + work["post_scf_full_builds"]
                                >= 2
                            )
                        if arm == "fast":
                            assert result.precision_validation is not None
                            assert result.precision_validation.profile == "fast"
                        records.append(
                            (
                                phase,
                                float(result.energy),
                                np.array(result.forces, copy=True),
                            )
                        )
                        if phase == "cold":
                            owner.set_warm_start_updates(False)
            finally:
                calculator.clear_cache()
            outcomes[choice] = records
    lib.num_threads(1)
    references = {}
    for geometry, xyz in (("original", coordinates), ("moved", moved)):
        mol = gto.M(
            atom=[(atom[0], point) for atom, point in zip(atoms, xyz, strict=True)],
            unit="Bohr",
            basis="def2-svp",
            cart=False,
            verbose=0,
        )
        reference = rks.RKS(mol)
        if mode == "df":
            reference = reference.density_fit()
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
        references[geometry] = (energy, -cp.asnumpy(gradient.kernel()))
    for choice, records in outcomes.items():
        assert len(records) == 5, choice
        for phase, actual_energy, actual_forces in records:
            energy, forces = references[
                "original" if phase in ("cold", "warm") else "moved"
            ]
            delta = actual_forces - forces
            assert abs(actual_energy - energy) / len(atoms) < 2.5e-6
            assert np.max(np.abs(delta)) < 2.5e-5
            assert np.sqrt(np.mean(delta**2)) < 7.5e-6


@pytest.mark.parametrize(
    "control", ("strict", "budget", "grid", "small", "legacy-off", "zero")
)
def test_unqualified_default_preserves_explicit_off(
    control: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise real preparation exclusions, including the active resource ledger.

    These paired native results protect fallback equivalence. Independent E/F
    accuracy is established separately, not inferred from agreement with off.
    """
    from generativeqc import Calculator, GridSpec, KsOptions, ResourceBudget

    from benchmarks.readme_hf_scaling import scaling_cases

    assert os.environ.get("SLURM_JOB_ID")
    atoms = scaling_cases()["water-3" if control == "small" else "water-12"].atoms
    options = {}
    if control == "strict":
        options.update(energy_tolerance=1e-12, density_tolerance=1e-9)
    elif control == "budget":
        options["resource_budget"] = ResourceBudget(
            host_bytes=16 << 30, device_bytes=16 << 30
        )
    elif control == "grid":
        options["ks_options"] = KsOptions(grid=GridSpec(24, 12, 24))
    results = []
    for choice in ("unset", "off"):
        with monkeypatch.context() as scope:
            scope.delenv("GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK", raising=False)
            scope.delenv("GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK", raising=False)
            if control == "legacy-off":
                scope.setenv("GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK", "off")
            if control == "zero":
                scope.setenv("GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK", "0")
            if choice == "off":
                scope.setenv("GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK", "off")
            calculator = Calculator(
                method="pbe0-rks",
                device="cuda",
                basis="def2-svp",
                basis_representation="spherical",
                **options,
            )
            try:
                with calculator.prepare_batch([atoms], warm_start=True) as owner:
                    item = owner.execute(
                        strict=True, properties=("energy", "forces")
                    ).items[0]
                    assert item.converged and item.executed_backend == "cuda"
                    assert not item.incremental_direct_jk["requested"]
                    assert not item.incremental_direct_jk["active"]
                    results.append(
                        (float(item.energy), np.array(item.forces, copy=True))
                    )
            finally:
                calculator.clear_cache()
    assert abs(results[0][0] - results[1][0]) < 1e-9
    np.testing.assert_allclose(results[0][1], results[1][1], atol=1e-8, rtol=0)


@pytest.mark.parametrize(
    "arm,mode", (("fast", "direct"), ("standard", "df"), ("fast", "df"))
)
def test_explicit_incremental_rejects_conflicting_provider(
    arm: str,
    mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automatic fallback must not weaken the explicit selector's error contract."""
    from benchmarks.readme_defaults import native_calculator, prepare_inputs
    from benchmarks.readme_hf_scaling import scaling_cases

    assert os.environ.get("SLURM_JOB_ID")
    monkeypatch.setenv("GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK", "on")
    monkeypatch.delenv("GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK", raising=False)
    if mode == "df":
        prepare_inputs(tmp_path)
    calculator = native_calculator("pbe0", arm, mode, tmp_path)
    try:
        with (
            pytest.raises((ValueError, RuntimeError), match="strict-FP64|exact-direct"),
            calculator.prepare_batch([scaling_cases()["water-12"].atoms]) as owner,
        ):
            owner.execute(strict=True, properties=("energy", "forces"))
    finally:
        calculator.clear_cache()


def test_imported_seed_does_not_claim_native_cold_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exported physical density stays external when restored to a new owner."""
    from generativeqc import _native
    from test_dft_batch import restore_snapshots, warm_snapshot

    from benchmarks.readme_defaults import native_calculator
    from benchmarks.readme_hf_scaling import scaling_cases

    assert os.environ.get("SLURM_JOB_ID")
    monkeypatch.delenv("GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK", raising=False)
    monkeypatch.delenv("GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK", raising=False)
    calculator = native_calculator("pbe0", "standard", "direct", tmp_path)
    atoms = scaling_cases()["water-12"].atoms
    try:
        with calculator.prepare_batch([atoms], warm_start=True) as original:
            cold = original.execute(strict=True, properties=("energy", "forces")).items[
                0
            ]
            assert cold.incremental_direct_jk["active"]
            snapshot = warm_snapshot(original, 0)
        with calculator.prepare_batch([atoms], warm_start=True) as restored:
            _native.check(restored._library, restore_snapshots(restored, [snapshot]))
            imported = restored.execute(
                strict=True, properties=("energy", "forces")
            ).items[0]
            assert imported.converged and imported.warm_start_used
            assert imported.incremental_direct_jk["requested"]
            assert not imported.incremental_direct_jk["active"]
            assert imported.incremental_direct_jk["delta_builds"] == 0
            assert abs(imported.energy - cold.energy) < 1e-8
            np.testing.assert_allclose(imported.forces, cold.forces, atol=2e-6, rtol=0)
    finally:
        calculator.clear_cache()
