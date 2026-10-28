"""Qualify default SCF error separately from quadrature and density fitting."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="requires a finite Slurm GPU allocation and independent GPU4PySCF",
)


@pytest.mark.parametrize(
    "method,mode,system",
    [
        (method, mode, "water")
        for method in ("hf", "pbe", "pbe0", "b3lyp", "r2scan", "r2scan-3c", "m06-2x")
        for mode in ("direct", "df")
    ]
    + [
        (method, "direct", system)
        for method in ("hf", "pbe0", "m06-2x")
        for system in ("ammonia", "methane")
    ],
)
def test_default_complete_endpoint_against_tight_same_model(
    method: str, mode: str, system: str, tmp_path: Path
) -> None:
    """Original/displaced cold and frozen warm calls use no reference seed.

    The independent oracle solves the same fitted operator and stock grid.
    Allocate at most a quarter of the common E/F budget to solver error; this
    does not qualify the separate default-grid or auxiliary-fitting error.
    """
    import cupy as cp
    from gpu4pyscf.dft import rks
    from gpu4pyscf.scf import hf
    from pyscf import gto, lib

    from benchmarks.readme_defaults import (
        METHODS,
        native_calculator,
        orbital_basis,
        prepare_inputs,
        reference_r2scan3c_correction,
    )

    assert os.environ.get("SLURM_JOB_ID")
    lib.num_threads(1)
    prepare_inputs(tmp_path)
    geometries = {
        "water": [
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, 1.8)),
            ("H", (1.7, 0.0, -0.6)),
        ],
        "ammonia": [
            ("N", (0.0, 0.0, 0.2)),
            ("H", (0.0, 1.8, -0.45)),
            ("H", (1.56, -0.9, -0.45)),
            ("H", (-1.56, -0.9, -0.45)),
        ],
        "methane": [
            ("C", (0.0, 0.0, 0.0)),
            ("H", (1.18, 1.18, 1.18)),
            ("H", (-1.18, -1.18, 1.18)),
            ("H", (1.18, -1.18, -1.18)),
            ("H", (-1.18, 1.18, -1.18)),
        ],
    }
    atoms = geometries[system]
    displaced = [
        (s, tuple(np.array(xyz) + ([0.02, 0, 0] if i == 2 else [0, 0, 0])))
        for i, (s, xyz) in enumerate(atoms)
    ]
    receipt = {"method": method, "mode": mode, "system": system, "samples": []}
    for geometry in (atoms, displaced):
        mol = gto.M(
            atom=geometry,
            unit="Bohr",
            basis=orbital_basis(method),
            cart=False,
            verbose=0,
        )
        reference = hf.RHF(mol) if method == "hf" else rks.RKS(mol)
        if method != "hf":
            reference.xc = METHODS[method][1]
        if mode == "df":
            reference = reference.density_fit()
        reference.conv_tol = 1e-12
        reference.conv_tol_grad = 1e-9
        reference.max_cycle = 150
        reference.direct_scf = False
        reference.kernel()
        reference_solver = "diis"
        if not reference.converged:
            # Symmetric M06-2X geometries can stall ordinary DIIS at these
            # deliberately tight oracle controls. Continue only the independent
            # reference with PySCF's CPU second-order solver, preserving the
            # model, stock grid and both thresholds. GPU Newton can stall too;
            # this is oracle work only, and no native seed enters the reference.
            density = reference.make_rdm1()
            reference = reference.to_cpu().newton()
            reference.kernel(dm0=cp.asnumpy(density))
            reference_solver = "gpu-diis+cpu-newton"
        assert reference.converged
        gradient = reference.nuc_grad_method()
        if method != "hf":
            gradient.grid_response = True
        energy = float(reference.e_tot)
        forces = -cp.asnumpy(gradient.kernel())
        if method == "r2scan-3c":
            correction_energy, correction_gradient = reference_r2scan3c_correction(
                geometry
            )()
            energy += float(correction_energy)
            forces -= correction_gradient
        calculator = native_calculator(method, "standard", mode, tmp_path)
        try:
            assert calculator._energy_tolerance == 1e-9
            assert calculator._density_tolerance == 1e-5
            with calculator.prepare_batch([geometry], warm_start=True) as owner:
                for repeat in range(2):
                    result = owner.execute(
                        properties=("energy", "forces"), strict=True
                    ).items[0]
                    error = result.forces - forces
                    checks = {
                        "energy_per_atom": abs(result.energy - energy) / len(atoms),
                        "force_max": float(np.max(np.abs(error))),
                        "force_rms": float(np.sqrt(np.mean(error**2))),
                    }
                    receipt["samples"].append(
                        {
                            "geometry": geometry,
                            "repeat": repeat,
                            "iterations": result.iterations,
                            "energy": result.energy,
                            "forces": result.forces.tolist(),
                            "reference_energy": energy,
                            "reference_forces": forces.tolist(),
                            "reference_solver": reference_solver,
                            "errors": checks,
                        }
                    )
                    assert result.converged and result.executed_backend == "cuda"
                    assert checks["energy_per_atom"] < 2.5e-6
                    assert checks["force_max"] < 2.5e-5
                    assert checks["force_rms"] < 7.5e-6
                    if repeat:
                        assert result.warm_start_used
                    owner.set_warm_start_updates(False)
        finally:
            calculator.clear_cache()
            if destination := os.environ.get("GENERATIVEQC_SCF_ORACLE_RECEIPTS"):
                output = Path(destination)
                output.mkdir(parents=True, exist_ok=True)
                (output / f"{method}-{mode}-{system}.json").write_text(
                    json.dumps(receipt, indent=2) + "\n"
                )
