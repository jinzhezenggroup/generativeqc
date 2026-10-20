"""Default-grid Fast warm reuse against independent moving-grid observables."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc import PrecisionCalculator

from benchmarks.readme_hf_scaling import scaling_cases

if TYPE_CHECKING:
    from pathlib import Path

    from generativeqc import BatchItemResult
    from generativeqc._precision_batch import PreparedPrecisionBatch

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires a finite Slurm GPU allocation",
)


def oracle(atoms: tuple, functional: str) -> tuple[float, np.ndarray]:
    """Independent CPU quadrature and full analytic moving-grid response."""
    from pyscf import dft, gto

    molecule = gto.M(atom=atoms, basis="def2-svp", unit="Bohr", verbose=0)
    engine = dft.RKS(molecule)
    engine.xc = functional
    engine.grids.level = 4
    engine.conv_tol = 1e-11
    engine.conv_tol_grad = 1e-7
    engine.direct_scf_tol = 1e-14
    energy = float(engine.kernel())
    assert engine.converged
    gradient = engine.nuc_grad_method()
    gradient.grid_response = True
    return energy, -gradient.kernel()


@pytest.mark.parametrize("functional", ("PBE", "PBE0", "R2SCAN", "M06-2X", "WB97M-V"))
def test_default_fast_frozen_and_moved_physical_endpoints(functional: str) -> None:
    """A strict warm replay retains E/F accuracy and cold/moved work remains live."""
    atoms = scaling_cases()["water-12"].atoms
    moved = tuple(
        (
            symbol,
            tuple(
                np.asarray(position) + ([0.003, -0.002, 0.001] if i == 1 else [0, 0, 0])
            ),
        )
        for i, (symbol, position) in enumerate(atoms)
    )
    calculator = PrecisionCalculator(
        policy="fast",
        method=functional.lower() + "-rks",
        basis="def2-svp",
        basis_representation="spherical",
        device="cuda",
    )
    try:
        with calculator.prepare_batch([atoms], warm_start=True) as prepared:
            for geometry_index, geometry in enumerate((atoms, moved)):
                energy, forces = oracle(geometry, functional)
                prepared.set_warm_start_updates(True)
                coordinates = (
                    None
                    if geometry_index == 0
                    else [np.asarray([p for _, p in geometry])]
                )
                for replay in range(3):
                    batch = prepared.execute(coordinates, strict=True)
                    observed = batch.items[0]
                    assert abs(observed.energy - energy) / len(atoms) < 1e-5
                    difference = observed.forces - forces
                    assert np.max(np.abs(difference)) < 1e-4
                    assert np.sqrt(np.mean(difference**2)) < 3e-5
                    assert observed.precision_validation.solve_count == 1
                    if replay:
                        assert observed.warm_start_used
                        assert observed.precision["mixed_stage_fock_builds"] == 0
                        assert observed.precision["effective_bits"] == 64
                        assert observed.precision["final_residual_audits"] > 0
                        assert (
                            observed.precision_validation.status
                            == "qualified_budget_fock_strict_warm"
                        )
                    else:
                        assert observed.precision["mixed_stage_fock_builds"] > 0
                        prepared.set_warm_start_updates(False)
                    # None always means the preparation geometry, not the last
                    # explicit geometry. Repeat the moved coordinates so this
                    # assertion compares forces for the same physical system.
    finally:
        calculator.clear_cache()


@pytest.mark.parametrize("invalidation", ("selection", "capacity", "buckets", "replay"))
def test_fast_range_warm_lease_and_policy_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, invalidation: str
) -> None:
    """Strict Fast replays consume existing values and retire changed policies.

    The independent frozen/moved oracle above guards physical observables. This
    test distinguishes real value reuse from an ordinary-provider fallback that
    happens to return the same energy. No warm call may rebuild quartic storage.
    """
    trace = tmp_path / "range.jsonl"
    monkeypatch.setenv("GENERATIVEQC_DF_PROGRESS_TRACE", str(trace))
    for key in (
        "GENERATIVEQC_KS_RANGE_RESIDENT_VALUES",
        "GENERATIVEQC_KS_RANGE_RESIDENT_REPLAY",
        "GENERATIVEQC_KS_RESIDENT_BUCKET_MASK",
        "GENERATIVEQC_KS_RESIDENT_MAX_BYTES",
    ):
        monkeypatch.delenv(key, raising=False)
    calculator = PrecisionCalculator(
        policy="fast",
        method="wb97m-v",
        basis="def2-svp",
        basis_representation="spherical",
        device="cuda",
    )
    offset = 0

    def execute(owner: PreparedPrecisionBatch) -> tuple[BatchItemResult, int, int]:
        nonlocal offset
        result = owner.execute(strict=True).items[0]
        lines = trace.read_text().splitlines()
        rows = [json.loads(line) for line in lines[offset:]]
        offset = len(lines)
        built = sum(
            row["value"] for row in rows if row.get("key") == "source_values_built"
        )
        consumed = sum(
            row["value"]
            for row in rows
            if row.get("key") == "phase_resident_fock_actions"
        )
        assert result.converged
        return result, built, consumed

    try:
        with calculator.prepare_batch(
            [scaling_cases()["water-12"].atoms], warm_start=True
        ) as owner:
            cold, built, consumed = execute(owner)
            assert built > 0 and consumed > 0
            assert cold.precision["mixed_stage_fock_builds"] > 0
            owner.set_warm_start_updates(False)
            warm, built, consumed = execute(owner)
            assert built == 0 and consumed > 0
            assert warm.precision["effective_bits"] == 64
            assert warm.precision["mixed_stage_fock_builds"] == 0
            assert warm.precision_validation.solve_count == 1
            key, value = {
                "selection": ("GENERATIVEQC_KS_RANGE_RESIDENT_VALUES", "0"),
                "capacity": ("GENERATIVEQC_KS_RESIDENT_MAX_BYTES", "1"),
                "buckets": ("GENERATIVEQC_KS_RESIDENT_BUCKET_MASK", "0x1"),
                "replay": ("GENERATIVEQC_KS_RANGE_RESIDENT_REPLAY", "0"),
            }[invalidation]
            monkeypatch.setenv(key, value)
            fallback, built, consumed = execute(owner)
            assert built == 0 and consumed == 0
            for result in (warm, fallback):
                assert abs(result.energy - cold.energy) / 12 < 1e-5
                difference = result.forces - cold.forces
                assert np.max(np.abs(difference)) < 1e-4
                assert np.sqrt(np.mean(difference**2)) < 3e-5
    finally:
        calculator.clear_cache()
