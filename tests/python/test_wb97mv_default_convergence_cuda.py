"""Independent E/F and warm-state gates for WB97M-V Standard stopping."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pytest
from generativeqc import Calculator

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="allocated CUDA WB97M-V default-convergence qualification",
)


@pytest.mark.parametrize("displacement", (0.0, 0.02))
def test_default_and_explicit_strict_complete_physical_endpoints(
    displacement: float,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use an independent same-grid solve; never lend its density to native SCF."""
    import cupy as cp
    from gpu4pyscf.dft import rks
    from pyscf import gto, lib

    from benchmarks.readme_defaults_accuracy import errors

    lib.num_threads(1)
    atoms = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 1.8)),
        ("H", (1.7 + displacement, 0.0, -0.6)),
    ]
    molecule = gto.M(atom=atoms, unit="Bohr", basis="def2-svp", cart=False, verbose=0)
    reference = rks.RKS(molecule)
    reference.xc = "WB97M_V"
    reference.conv_tol = 1e-12
    reference.conv_tol_grad = 1e-9
    reference.max_cycle = 150
    reference.kernel()
    assert reference.converged
    gradient = reference.nuc_grad_method()
    gradient.grid_response = True
    oracle = {
        "energy": float(reference.e_tot),
        "forces": -cp.asnumpy(gradient.kernel()),
    }

    for controls in ({}, {"energy_tolerance": 1e-10, "density_tolerance": 1e-8}):
        trace = tmp_path / ("strict.jsonl" if controls else "default.jsonl")
        monkeypatch.setenv("GENERATIVEQC_DF_PROGRESS_TRACE", str(trace))
        calculator = Calculator(
            method="wb97m-v-rks",
            device="cuda",
            basis="def2-svp",
            basis_representation="spherical",
            density_fitting="none",
            **controls,
        )
        assert calculator._energy_tolerance == (1e-10 if controls else 1e-9)
        assert calculator._density_tolerance == (1e-8 if controls else 1e-5)
        owner = calculator.prepare_batch([atoms], warm_start=True)
        try:
            for repeat in range(2):
                result = owner.execute(
                    strict=True, properties=("energy", "forces")
                ).items[0]
                assert result.converged and result.executed_backend == "cuda"
                assert np.isfinite(result.forces).all()
                assert errors(result.energy, result.forces, oracle, len(atoms))[
                    "passed"
                ]
                assert result.warm_start_used == bool(repeat)
                if not repeat:
                    owner.set_warm_start_updates(False)
        finally:
            owner.close()
            calculator.clear_cache()
        # Scientific parity alone can pass through the strict fallback. Prove
        # the qualified default actually uses its physical MO-gradient gate.
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        checks = sum(
            row.get("value", 0)
            for row in rows
            if row.get("key") == "physical_orbital_gradient_checks"
        )
        assert checks == 0 if controls else checks > 0
