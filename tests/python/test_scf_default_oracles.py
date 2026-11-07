"""Independent endpoint oracles for the shared public convergence defaults."""

from __future__ import annotations

import os

import numpy as np
import pytest
from generativeqc import Calculator


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
                reason="requires an explicitly enabled finite Slurm GPU allocation",
            ),
        ),
    ],
)
@pytest.mark.parametrize("method", ["rhf", "uhf", "pbe-rks", "pbe-uks"])
def test_default_cold_warm_and_displaced_endpoints_against_pyscf(
    method: str, device: str
) -> None:
    """Compare complete native E/F against an independently tight same-model solve.

    Neutral water and its unrestricted cation exercise distinct physical owners. A
    displaced geometry prevents accepting a stale warm energy or force. DFT uses
    the shared stock Treutler/Becke grid on both sides, including grid response.
    The reference is evaluated only in the test and never seeds native execution.
    """
    pytest.importorskip("pyscf")
    from pyscf import dft, gto, lib, scf

    if device == "cuda":
        assert os.environ.get("SLURM_JOB_ID")
    unrestricted = method in {"uhf", "pbe-uks"}
    atoms = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 1.8)),
        ("H", (1.7, 0.0, -0.6)),
    ]
    charge = 1 if unrestricted else 0
    multiplicity = 2 if unrestricted else 1
    previous_threads = lib.num_threads()
    lib.num_threads(1)
    calculator = Calculator(method=method, basis="sto-3g", device=device)
    try:
        assert calculator._energy_tolerance == scf.hf.SCF.conv_tol == 1e-9
        assert calculator._density_tolerance == 1e-5
        for displacement in (0.0, 0.025):
            geometry = [
                (symbol, (position[0] + displacement, *position[1:]))
                if index == 1
                else (symbol, position)
                for index, (symbol, position) in enumerate(atoms)
            ]
            mol = gto.M(
                atom=geometry,
                basis="sto-3g",
                unit="Bohr",
                spin=multiplicity - 1,
                charge=charge,
                cart=True,
                verbose=0,
            )
            if method.startswith("pbe"):
                reference = dft.UKS(mol) if unrestricted else dft.RKS(mol)
                reference.xc = "pbe"
            else:
                reference = scf.UHF(mol) if unrestricted else scf.RHF(mol)
            reference.conv_tol = 1e-12
            reference.conv_tol_grad = 1e-9
            reference.max_cycle = 150
            reference.direct_scf = False
            reference.kernel()
            if not reference.converged:
                reference = reference.newton()
                reference.kernel(dm0=reference.make_rdm1())
            assert reference.converged
            gradient = reference.nuc_grad_method()
            if method.startswith("pbe"):
                gradient.grid_response = True
            reference_forces = -gradient.kernel()
            with calculator.prepare_batch(
                [geometry],
                charges=[charge],
                multiplicities=[multiplicity],
                warm_start=True,
            ) as owner:
                for _repeat in range(2):
                    result = owner.execute(
                        properties=("energy", "forces"), strict=True
                    ).items[0]
                    backend = "cpu_reference" if device == "cpu" else "cuda"
                    assert result.converged and result.executed_backend == backend
                    if _repeat:
                        assert result.warm_start_used
                    assert abs(result.energy - reference.e_tot) < 1e-7
                    np.testing.assert_allclose(
                        result.forces, reference_forces, atol=2e-6, rtol=0
                    )
    finally:
        calculator.clear_cache()
        lib.num_threads(previous_threads)
