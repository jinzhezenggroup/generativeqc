"""Complete public HF/KS DF with the reference's unmodified default fit basis.

PySCF independently supplies metric/three-center integrals, SCF and all nuclear
derivatives. Default Treutler grids remain intact; the oracle explicitly includes
moving-grid response to test the native force definition. Real CUDA execution
requires a finite Slurm allocation. These are correctness gates, not timings.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, PrecisionCalculator, load_r2scan3c_basis
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc.elements import SYMBOLS

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DF_G_ENDPOINT_TEST") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires an explicit finite Slurm CUDA allocation",
)

METHODS = (
    ("rhf", None),
    ("pbe-rks", "PBE"),
    ("pbe0-rks", "PBE0"),
    ("b3lyp-rks", "B3LYP"),
    ("r2scan-rks", "R2SCAN"),
    ("m06-2x-rks", "M06-2X"),
    ("wb97m-v-rks", "WB97M_V"),
    ("r2scan-3c", "R2SCAN"),
    ("uhf", None),
    ("pbe-uks", "PBE"),
)


def auxiliary_snapshot(molecule: Any) -> BasisSet:
    """Transfer only basis data from the default independent reference owner."""
    import pyscf
    from pyscf import df, gto

    tables = df.addons.make_auxmol(molecule, None)._basis
    assert max(s[0] for shells in tables.values() for s in shells) == 4
    return BasisSet(
        name="reference-default-jkfit",
        elements=tuple(
            ElementBasis(
                gto.charge(symbol),
                tuple(
                    BasisShell(
                        angular_momentum=shell[0],
                        exponents=tuple(str(row[0]) for row in shell[1:]),
                        coefficients=tuple(
                            tuple(str(row[col]) for row in shell[1:])
                            for col in range(1, len(shell[1]))
                        ),
                    )
                    for shell in shells
                ),
            )
            for symbol, shells in sorted(tables.items())
        ),
        representation="spherical",
        provenance=BasisProvenance(
            "PySCF default auxiliary tables",
            pyscf.__version__,
            "Apache-2.0",
            hashlib.sha256(json.dumps(tables, sort_keys=True).encode()).hexdigest(),
        ),
    )


def check_endpoint(
    method: str,
    xc: str | None,
    fast: bool,
    case_name: str,
    *,
    density_fitting: str = "cuda",
) -> None:
    """Exercise public preparation, metric response and retained warm ownership."""
    from pyscf import dft, gto, lib, scf
    from test_dft_complete_cuda import no_cpu_derivatives

    from benchmarks._cases import benchmark_cases

    lib.num_threads(1)
    atoms = list(benchmark_cases()[case_name].atoms)
    open_shell = method in ("uhf", "pbe-uks")
    if open_shell:
        atoms = atoms[:2]
    basis: Any = "def2-svp"
    if method == "r2scan-3c":
        basis = {
            SYMBOLS[element.atomic_number - 1]: [
                [
                    shell.angular_momentum,
                    *[
                        [float(exponent), *[float(c[i]) for c in shell.coefficients]]
                        for i, exponent in enumerate(shell.exponents)
                    ],
                ]
                for shell in element.shells
            ]
            for element in load_r2scan3c_basis().elements
        }
    mol = gto.M(atom=atoms, unit="Bohr", basis=basis, spin=int(open_shell), verbose=0)
    auxiliary = auxiliary_snapshot(mol)
    solver = (
        (dft.UKS if open_shell else dft.RKS)(mol)
        if xc
        else (scf.UHF if open_shell else scf.RHF)(mol)
    ).density_fit()
    if method == "wb97m-v-rks":
        # Native DF-RSH fits full-range J/K but retains exact long-range K.
        # Match that operator in the independent oracle; the stock reference
        # benchmark remains fully fitted and is not an accuracy oracle here.
        from gpu4pyscf.dft import rks as gpu_rks

        from benchmarks.readme_wb97mv import configure_reference_mixed_rsh_df

        solver = gpu_rks.RKS(mol).density_fit()
        solver.xc = xc
        configure_reference_mixed_rsh_df(solver)
    if xc:
        solver.xc = xc
    solver.conv_tol, solver.conv_tol_grad = 1e-12, 1e-9
    expected_energy = solver.kernel()
    assert solver.converged
    gradient = solver.nuc_grad_method()
    gradient.auxbasis_response = True
    if xc:
        gradient.grid_response = True
    expected_forces = -gradient.kernel()
    if hasattr(expected_forces, "get"):
        expected_forces = expected_forces.get()
    if method == "r2scan-3c":
        from benchmarks.readme_omol25 import reference_correction

        energy, derivative = reference_correction(atoms)()
        expected_energy += energy
        expected_forces -= derivative
    options: dict[str, Any] = {
        "method": method,
        "device": "cuda",
        "basis_representation": "spherical",
        "auxiliary_basis": auxiliary,
        "density_fitting": density_fitting,
    }
    if method != "r2scan-3c":
        options["basis"] = "def2-svp"
    calculator = (
        PrecisionCalculator(policy="fast", **options) if fast else Calculator(**options)
    )
    try:
        with calculator.prepare_batch(
            [atoms], multiplicities=[2 if open_shell else 1], warm_start=True
        ) as batch:
            for phase in range(2):
                with no_cpu_derivatives():
                    result = batch.execute(
                        strict=True, properties=("energy", "forces")
                    ).items[0]
                assert result.converged and result.executed_backend == "cuda"
                assert abs(result.energy - expected_energy) < (1e-5 if fast else 2e-7)
                delta = result.forces - expected_forces
                assert np.max(np.abs(delta)) < (1e-4 if fast else 2e-6)
                assert np.sqrt(np.mean(delta**2)) < (3e-5 if fast else 7e-7)
                np.testing.assert_allclose(
                    result.forces.sum(axis=0), 0, atol=2e-8, rtol=0
                )
                if phase:
                    assert result.warm_start_used
                else:
                    batch.set_warm_start_updates(False)
    finally:
        calculator.clear_cache()


@pytest.mark.parametrize("method,xc", METHODS)
@pytest.mark.parametrize("fast", (False, True))
def test_default_auxiliary_complete_cold_and_warm(
    method: str, xc: str | None, fast: bool
) -> None:
    check_endpoint(method, xc, fast, "water-def2-svp-spherical")


@pytest.mark.parametrize("method,xc", (("rhf", None), ("pbe-rks", "PBE")))
@pytest.mark.parametrize("fast", (False, True))
def test_default_auxiliary_water_tetramer(
    method: str, xc: str | None, fast: bool
) -> None:
    """A larger complete owner exercises auxiliary panels and repeated response."""
    check_endpoint(method, xc, fast, "water-tetramer-def2-svp-spherical")


@pytest.mark.parametrize("schedule", ("warp", "packed", "compact"))
@pytest.mark.parametrize("pairs", ("full", "symmetric"))
def test_auxiliary_g_shell_schedules_against_independent_forces(
    schedule: str, pairs: str, monkeypatch: Any, tmp_path: Any
) -> None:
    """Exercise every schedule on spherical g weights and both J/K responses.

    The independent complete PySCF derivative includes auxiliary-center motion.
    Executed g-shell regions ensure agreement cannot come from generic replay.
    Small panels naturally cut spherical shells in the exchange response.
    """
    monkeypatch.setenv("GENERATIVEQC_DF_WEIGHTED_EXECUTION", "shell")
    monkeypatch.setenv("GENERATIVEQC_DF_SHELL_SCHEDULE", schedule)
    monkeypatch.setenv("GENERATIVEQC_DF_DERIVATIVE_PAIRS", pairs)
    trace = tmp_path / "g-shell.jsonl"
    monkeypatch.setenv("GENERATIVEQC_DF_TRACE", str(trace))
    check_endpoint("pbe0-rks", "PBE0", False, "water-def2-svp-spherical")
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    import re

    assert any(
        re.fullmatch(r"shell_[0-3][0-3]4(?:_p\d+_\d+_\d+)?", region["name"])
        for record in records
        for region in record["regions"]
    )
