"""Default streamed DF SCF/forces, frozen replay and changed-geometry ownership."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pytest
from generativeqc import Calculator, PrecisionCalculator

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1"
    or not (os.environ.get("SLURM_JOB_ID") or os.environ.get("LSB_JOBID")),
    reason="requires an explicitly allocated finite Slurm/LSF GPU job",
)


@pytest.fixture(
    scope="module", params=[("PBE", False), ("PBE", True), ("R2SCAN", False)]
)
def endpoint_oracles(request: Any) -> dict:
    """Independent tight DF equations and full grid/auxiliary force response."""
    import cupy as cp
    from gpu4pyscf.dft import rks
    from pyscf import df, gto, lib
    from test_scf_df_auxiliary_g_cuda import auxiliary_snapshot

    from benchmarks._cases import benchmark_cases

    lib.num_threads(1)
    functional, cartesian = request.param
    atoms = benchmark_cases()["water-tetramer-def2-svp-spherical"].atoms
    moved = [
        (symbol, tuple(np.asarray(xyz) + (0, 0, 0.025) if i == 1 else xyz))
        for i, (symbol, xyz) in enumerate(atoms)
    ]
    rows = []
    for geometry in (atoms, moved):
        molecule = gto.M(
            atom=geometry, unit="Bohr", basis="def2-svp", cart=cartesian, verbose=0
        )
        reference = rks.RKS(molecule).density_fit()
        # The frozen auxiliary snapshot is spherical, independently of the
        # orbital representation. Match that physical model explicitly; a
        # Cartesian auxiliary adds trace functions and a different metric rank.
        reference.with_df.auxmol = df.addons.make_auxmol(molecule, None)
        reference.with_df.auxmol.cart = False
        reference.xc = functional
        reference.conv_tol, reference.conv_tol_grad = 1e-12, 1e-9
        reference.max_cycle = 150
        reference.kernel()
        assert reference.converged
        gradient = reference.nuc_grad_method()
        gradient.grid_response = gradient.auxbasis_response = True
        rows.append((float(reference.e_tot), -cp.asnumpy(gradient.kernel())))
    return {
        "functional": functional,
        "cartesian": cartesian,
        "geometries": (atoms, moved),
        "auxiliary": auxiliary_snapshot(molecule),
        "references": rows,
    }


@pytest.mark.parametrize("rollback", (False, True), ids=("default", "raw-fallback"))
@pytest.mark.parametrize("fast", (False, True), ids=("standard", "fast"))
@pytest.mark.parametrize(
    "tight", (False, True), ids=("native-controls", "explicit-tight")
)
def test_default_df_force_geometry_and_frozen_replay(
    endpoint_oracles: dict,
    rollback: bool,
    fast: bool,
    tight: bool,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A moved atom must invalidate both value/metric and derivative ownership.

    The cap deliberately requires streaming; the explicit zero selector keeps
    the same complete raw fallback qualified. Neither reference densities nor
    reference integrals enter the production solver. This is a correctness
    test, not a moved-geometry benchmark population.
    """
    from test_dft_complete_cuda import no_cpu_derivatives

    if rollback:
        monkeypatch.setenv("GENERATIVEQC_DF_MD_COULOMB", "0")
    else:
        monkeypatch.delenv("GENERATIVEQC_DF_MD_COULOMB", raising=False)
    monkeypatch.setenv("GENERATIVEQC_DF_VALUE_STORAGE", "dense")
    data = endpoint_oracles
    options = {
        "method": data["functional"].lower() + "-rks",
        "basis": "def2-svp",
        "basis_representation": "cartesian" if data["cartesian"] else "spherical",
        "device": "cuda",
        "auxiliary_basis": data["auxiliary"],
        "density_fitting": "cuda",
        "density_fitting_memory_budget_bytes": 64 << 20,
    }
    if tight:
        options.update(
            energy_tolerance=1e-12, density_tolerance=1e-10, screening_tolerance=1e-14
        )
    calculator = (
        PrecisionCalculator(policy="fast", **options) if fast else Calculator(**options)
    )
    try:
        with calculator.prepare_batch(
            [data["geometries"][0]], warm_start=True
        ) as batch:
            for index, (geometry, reference) in enumerate(
                zip(data["geometries"], data["references"], strict=True)
            ):
                trace = tmp_path / f"geometry-{index}.jsonl"
                monkeypatch.setenv("GENERATIVEQC_DF_TRACE", str(trace))
                batch.set_warm_start_updates(True)
                for repeat in range(2):
                    with no_cpu_derivatives():
                        result = batch.execute(
                            None
                            if not index
                            else [np.asarray([xyz for _, xyz in geometry])],
                            properties=("energy", "forces"),
                            strict=True,
                        ).items[0]
                    assert result.converged and result.executed_backend == "cuda"
                    assert abs(result.energy - reference[0]) < (
                        1e-5 * len(geometry) if fast else 2e-7
                    )
                    difference = result.forces - reference[1]
                    assert np.max(abs(difference)) < (1e-4 if fast else 2e-6)
                    assert np.sqrt(np.mean(difference**2)) < (3e-5 if fast else 7e-7)
                    if fast:
                        validation = result.precision_validation
                        assert validation.solve_count == 1
                        assert validation.status.startswith("qualified_")
                        for solve in validation.solves:
                            if tight:
                                assert solve.energy_tolerance <= 1e-12
                                assert solve.density_tolerance <= 1e-10
                    if repeat:
                        assert result.warm_start_used
                    batch.set_warm_start_updates(False)
                records = [json.loads(line) for line in trace.read_text().splitlines()]
                # Prove that this numerical gate exercised the selected
                # consumer; do not pin its internal tile or exact call counts.
                assert any(
                    row["counters"].get(
                        "response_raw_coulomb_source_passes"
                        if rollback
                        else "response_md_derivative_calls"
                    )
                    for row in records
                )
    finally:
        calculator.clear_cache()
