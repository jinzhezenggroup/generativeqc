"""Fast source capacity refusal and independent batched observable gates."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc import Calculator, PrecisionCalculator, ResourceBudget

from benchmarks._cases import benchmark_cases

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires a finite Slurm GPU allocation",
)


@pytest.fixture(scope="module")
def independent_geometries() -> tuple:
    """Converge both geometries independently; never export a native seed."""
    from pyscf import dft, gto, lib

    lib.num_threads(1)
    atoms = benchmark_cases()["water-tetramer-def2-svp-spherical"].atoms
    moved = tuple(
        (symbol, tuple(np.asarray(point) + ([0.003, -0.002, 0.001] if i == 1 else 0)))
        for i, (symbol, point) in enumerate(atoms)
    )
    references = []
    for geometry in (atoms, moved):
        mol = gto.M(atom=geometry, basis="def2-svp", unit="Bohr", verbose=0)
        reference = dft.RKS(mol)
        reference.xc = "pbe0"
        reference.conv_tol, reference.conv_tol_grad = 1e-12, 1e-8
        reference.direct_scf = False
        energy = float(reference.kernel())
        assert reference.converged
        gradient = reference.nuc_grad_method()
        gradient.grid_response = True
        references.append((energy, -gradient.kernel()))
    return (atoms, moved), references


@pytest.mark.parametrize("resource_mode", ("normal", "source-cap", "ledger"))
def test_fast_batch_source_capacity_and_reset(
    independent_geometries: tuple,
    resource_mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused optional source keeps both native owners numerically valid."""
    systems, references = independent_geometries
    trace = tmp_path / "source.jsonl"
    monkeypatch.setenv("GENERATIVEQC_DF_PROGRESS_TRACE", str(trace))
    monkeypatch.delenv("GENERATIVEQC_KS_MIXED_SOURCE_COARSE_SCREENING", raising=False)
    monkeypatch.delenv("GENERATIVEQC_KS_RESIDENT_VALUES", raising=False)
    monkeypatch.delenv("GENERATIVEQC_KS_RESIDENT_MAX_BYTES", raising=False)
    options = {}
    if resource_mode == "source-cap":
        monkeypatch.setenv("GENERATIVEQC_KS_RESIDENT_MAX_BYTES", "1")
    if resource_mode == "ledger":
        options["resource_budget"] = ResourceBudget(
            host_bytes=16 << 30, device_bytes=16 << 30
        )
    calculator = PrecisionCalculator(
        policy="fast",
        method="pbe0-rks",
        basis="def2-svp",
        basis_representation="spherical",
        device="cuda",
        **options,
    )
    try:
        with calculator.prepare_batch(systems, warm_start=True) as owner:
            for phase in ("cold", "warm", "reset"):
                if phase == "reset":
                    owner.clear_warm_starts()
                result = owner.execute(strict=True, properties=("energy", "forces"))
                for geometry, observed, (energy, forces) in zip(
                    systems, result.items, references, strict=True
                ):
                    assert observed.converged and observed.executed_backend == "cuda"
                    assert abs(observed.energy - energy) / len(geometry) < 2.5e-6
                    delta = observed.forces - forces
                    assert np.max(np.abs(delta)) < 2.5e-5
                    assert np.sqrt(np.mean(delta**2)) < 7.5e-6
                    assert observed.precision_validation.solve_count == 1
                    assert observed.precision["final_residual_audits"] > 0
                    if phase == "warm":
                        assert observed.warm_start_used
                        assert observed.precision["mixed_stage_fock_builds"] == 0
                if phase == "cold":
                    owner.set_warm_start_updates(False)
    finally:
        calculator.clear_cache()
    if resource_mode != "normal":
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        assert (
            sum(row["value"] for row in rows if row.get("key") == "source_values_built")
            == 0
        )


def test_invalid_source_selection_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed policy cannot silently select a different implementation."""
    monkeypatch.setenv("GENERATIVEQC_KS_MIXED_SOURCE_COARSE_SCREENING", "invalid")
    atoms = benchmark_cases()["water-tetramer-def2-svp-spherical"].atoms
    calculator = Calculator(
        method="pbe0-rks",
        basis="def2-svp",
        basis_representation="spherical",
        device="cuda",
    )
    try:
        with (
            pytest.raises(
                (ValueError, RuntimeError), match="MIXED_SOURCE_COARSE_SCREENING"
            ),
            calculator.prepare_batch([atoms]) as owner,
        ):
            owner.execute(strict=True)
    finally:
        calculator.clear_cache()
