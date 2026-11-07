"""Independent complete E/F and bounded shared stationary-panel qualification."""

from __future__ import annotations

import ctypes
import json
import os
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TEST_WB97MV_CUDA") != "1",
    reason="explicit scheduled CUDA shared-grid qualification required",
)


@pytest.mark.parametrize(
    "scenario", ("shared", "disabled", "minimum", "distinct", "moved")
)
def test_shared_panel_preserves_complete_force_and_fallbacks(
    scenario: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each model keeps its own moving quadrature and complete physical response.

    A minimum-capacity owner needs no optional resident panel. Distinct grids
    must retain both traversals, even when the selector explicitly requests
    reuse. Frozen warm seeds exercise both pair production and exact-key replay.
    """
    if not (os.environ.get("SLURM_JOB_ID") or os.environ.get("LSB_JOBID")):
        pytest.fail("finite scheduler allocation required")
    import cupy as cp
    from generativeqc import Calculator, GridSpec, KsOptions, _native
    from generativeqc_compiler.dft.nonlocal_policy import (
        MOLECULAR_VV10_DENSITY_THRESHOLD,
    )

    from benchmarks.readme_wb97mv import (
        reference_engine,
        reference_sample,
        reference_vv10_domain,
    )

    atoms = [
        ("O", (0.02, -0.03, 0.04)),
        ("H", (0.1, 1.43, 1.12 if scenario == "moved" else 1.1)),
        ("H", (-0.15, -1.45, 1.12)),
    ]
    primary = GridSpec(12, 4, 8)
    secondary = GridSpec(10, 4, 8) if scenario == "distinct" else primary
    engine = reference_engine(
        atoms, "def2-svp", primary, nonlocal_spec=secondary, full_fock=True
    )
    engine.conv_tol, engine.conv_tol_grad = 1e-12, 1e-10
    engine.direct_scf_tol = 1e-14
    with reference_vv10_domain(float(MOLECULAR_VV10_DENSITY_THRESHOLD)):
        expected = reference_sample(engine, cp)
    monkeypatch.setenv("GENERATIVEQC_VV10_COMPONENT_REUSE", "1")
    monkeypatch.setenv(
        "GENERATIVEQC_STATIONARY_SHARED_GRID", "0" if scenario == "disabled" else "1"
    )
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv("GENERATIVEQC_DF_TRACE", str(trace))
    calculator = Calculator(
        method="wb97m-v",
        basis="def2-svp",
        basis_representation="spherical",
        device="cuda",
        precision="fp64",
        density_fitting="none",
        ks_options=KsOptions(grid=primary, nonlocal_grid=secondary),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
        max_iterations=100,
    )
    evidence = []
    try:
        with calculator.prepare_batch([atoms], warm_start=True) as owner:
            assert owner._native_dft_force_eligible(0)
            if scenario == "minimum":
                query = owner._library.generativeqc_ks_batch_stationary_cuda_capacity_v1
                query.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.c_size_t,
                ]
                query.restype = ctypes.c_int
                capacity = (ctypes.c_uint64 * 9)()
                _native.check(
                    owner._library, query(owner._batch, 0, capacity, len(capacity))
                )
                limit = owner._library.generativeqc_ks_batch_stationary_cuda_limit_v1
                limit.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64]
                limit.restype = ctypes.c_int
                _native.check(owner._library, limit(owner._batch, 0, capacity[4]))
            for repeat in range(2):
                result = owner.execute(
                    strict=True, properties=("energy", "forces")
                ).items[0]
                assert result.converged and result.executed_backend == "cuda"
                assert abs(result.energy - expected["energies_hartree"][0]) < 2e-8
                np.testing.assert_allclose(
                    result.forces,
                    expected["forces_hartree_per_bohr"][0],
                    atol=2e-7,
                    rtol=0,
                )
                rows = [json.loads(line) for line in trace.read_text().splitlines()]
                work = [
                    row["counters"]
                    for row in rows
                    if row["operation"] == "stationary_force"
                ][-1]
                shared = scenario not in ("disabled", "distinct")
                assert bool(work["shared_nonlocal_grid"]) == shared
                primary_points = len(atoms) * 12 * 4 * 8
                secondary_points = len(atoms) * secondary.radial_points * 4 * 8
                assert work["grid_evaluation_points"] == primary_points + (
                    0 if shared else secondary_points
                )
                assert work["nonlocal_pair_sweeps"] == 1 - repeat
                assert work["nonlocal_pair_cache_hits"] == repeat
                evidence.append(
                    {
                        "energy": result.energy,
                        "forces": result.forces.tolist(),
                        "work": work,
                    }
                )
                owner.set_warm_start_updates(False)
                trace.write_text("")
    finally:
        calculator.clear_cache()
        (tmp_path / "results.json").write_text(json.dumps(evidence, indent=2) + "\n")
