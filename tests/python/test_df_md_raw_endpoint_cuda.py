"""Complete resident DF endpoints must preserve their independent E/F contract."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc import Calculator

from benchmarks.df_component_ledger import read_trace

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.mark.parametrize("method", ("rhf", "pbe0-rks"))
def test_prepared_and_generic_raw_complete_endpoints(
    method: str,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Independent forces on cold/warm/moved geometry and an explicit fallback.

    Trace selection proves the test actually exercised prepared E. It does not
    constrain launch counts, internal function names or recurrence spelling.
    """
    from pyscf import dft, gto, scf

    atoms = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.8)), ("H", (1.5, 0.0, -0.5))]
    moved = list(atoms)
    moved[1] = ("H", (0.003, 0.0, 1.801))
    references = []
    for geometry in (atoms, moved):
        molecule = gto.M(atom=geometry, unit="Bohr", basis="def2-svp", verbose=0)
        reference = (
            scf.RHF(molecule) if method == "rhf" else dft.RKS(molecule, xc="PBE0")
        )
        reference = reference.density_fit(auxbasis="def2-svp")
        reference.conv_tol, reference.conv_tol_grad = 1e-12, 1e-9
        reference.kernel()
        assert reference.converged
        gradient = reference.nuc_grad_method()
        if method != "rhf":
            gradient.grid_response = True
        references.append((reference.e_tot, -gradient.kernel()))
    monkeypatch.setenv("GENERATIVEQC_DF_VALUE_STORAGE", "packed-single")
    for mapping in ("auto", "auxiliary"):
        monkeypatch.setenv("GENERATIVEQC_DF_VALUE_MAPPING", mapping)
        # KS prepares its physical source in prepare_batch; HF may defer it
        # until execute. Observe the same complete cold boundary for both.
        monkeypatch.setenv(
            "GENERATIVEQC_DF_TRACE", str(tmp_path / f"{mapping}-0.jsonl")
        )
        calculator = Calculator(
            device="cuda",
            method=method,
            basis="def2-svp",
            basis_representation="spherical",
            auxiliary_basis="def2-svp",
            density_fitting="cuda",
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
        )
        with calculator.prepare_batch([atoms], warm_start=True) as owner:
            for index, geometry in enumerate((None, None, moved)):
                trace = tmp_path / f"{mapping}-{index}.jsonl"
                monkeypatch.setenv("GENERATIVEQC_DF_TRACE", str(trace))
                result = owner.execute(
                    [np.asarray([xyz for _, xyz in geometry])] if geometry else None,
                    strict=True,
                ).items[0]
                energy, forces = references[int(index == 2)]
                assert result.energy == pytest.approx(energy, rel=0, abs=1e-8)
                np.testing.assert_allclose(result.forces, forces, rtol=0, atol=3e-7)
                if index != 1:
                    records = read_trace(trace)
                    prepared = sum(
                        record["counters"].get("raw_hermite_values", 0)
                        for record in records
                    )
                    assert (prepared > 0) == (mapping == "auto")
