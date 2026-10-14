"""Large complete fitted candidates, independently of public size admission.

The private arithmetic request deliberately exercises the existing consumer
outside the qualified public domain. Passing these gates is necessary, not
sufficient, for widening that domain or claiming an endpoint speedup.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec
from generativeqc import _stationary_cuda as runtime
from test_df_fast_public_cuda import auxiliary_bases, options
from test_dft_complete_cuda import no_cpu_derivatives

from benchmarks._cases import benchmark_cases
from benchmarks.pbe0_precision import reference

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires a finite Slurm GPU allocation",
)
CASE = "water-hexadecamer-2s4-def2-svp-spherical"
GRID = GridSpec(48, 16, 32)


@pytest.mark.parametrize("functional", ("PBE", "PBE0", "B3LYP", "R2SCAN", "M06-2X"))
def test_large_fitted_candidate_cold_frozen_moved_and_strict_return(
    functional: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate every cold/replay force against an independently relaxed DF model.

    Strict and mixed owners each start cold, then retain only their own seed.
    Metric response, moving quadrature and all integral work remain enabled;
    no reference density or derivative is consumed by the production endpoint.
    """
    pytest.importorskip("gpu4pyscf")
    case = benchmark_cases()[CASE]
    assert case.expected_ao_count == 384 and len(case.atoms) == 48
    moved = [(symbol, np.asarray(position).copy()) for symbol, position in case.atoms]
    moved[1][1][0] += 0.025
    _, auxiliary = auxiliary_bases(case)
    oracles = [
        reference(
            CASE,
            GRID,
            4,
            functional=functional,
            atoms=atoms,
            auxiliary_basis=auxiliary,
            backend="gpu4pyscf",
        )
        for atoms in (case.atoms, moved)
    ]
    selected = False
    records = []
    original = runtime.complete_rks_cuda_gradient_diagnostic

    def execute(*args: Any, **kwargs: Any) -> Any:
        kwargs["fp32_semilocal_geometry"] = selected
        result = original(*args, **kwargs)
        records.append(result.work)
        return result

    monkeypatch.setattr(runtime, "complete_rks_cuda_gradient_diagnostic", execute)
    strict_forces = []
    for mixed in (False, True):
        selected = mixed
        calculator = Calculator(**options(CASE, functional, GRID), precision="fp64")
        try:
            with calculator.prepare_batch([case.atoms], warm_start=True) as batch:
                for geometry_index, (geometry, oracle) in enumerate(
                    zip(
                        (None, [np.asarray([position for _, position in moved])]),
                        oracles,
                        strict=True,
                    )
                ):
                    for frozen in (False, True):
                        batch.set_warm_start_updates(not frozen)
                        with no_cpu_derivatives():
                            result = batch.execute(geometry, strict=True).items[0]
                        assert result.converged and result.executed_backend == "cuda"
                        assert abs(result.energy - oracle["energy"]) < 1e-7
                        difference = result.forces - np.asarray(oracle["forces"])
                        assert np.max(np.abs(difference)) < (1e-5 if mixed else 1e-6)
                        assert np.sqrt(np.mean(difference**2)) < 3e-6
                        np.testing.assert_allclose(
                            result.forces.sum(axis=0), 0, atol=2e-8, rtol=0
                        )
                        if mixed:
                            baseline = strict_forces[2 * geometry_index + int(frozen)]
                            assert (
                                np.sqrt(np.mean((result.forces - baseline) ** 2)) < 3e-6
                            )
                        else:
                            strict_forces.append(result.forces.copy())
                        work = records[-1]
                        assert work["fp32_geometry_points"] == (
                            oracle["grid_points"] if mixed else 0
                        )
                        assert work["fp64_geometry_retries"] == 0
                        assert (
                            work["native_integral_resources"]["density_fitted_provider"]
                            == 1
                        )
                        if frozen:
                            assert result.warm_start_used
                if mixed:
                    selected = False
                    with no_cpu_derivatives():
                        replay = batch.execute(geometry, strict=True).items[0]
                    assert records[-1]["fp32_geometry_points"] == 0
                    np.testing.assert_allclose(
                        replay.forces, strict_forces[-1], atol=1e-7, rtol=0
                    )
        finally:
            calculator.clear_cache()
