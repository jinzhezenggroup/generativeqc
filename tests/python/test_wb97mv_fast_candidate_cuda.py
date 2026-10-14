"""Diagnostic full-composite candidate; not public WB97M-V Fast admission."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc._stationary_composite_cuda import (
    PreparedCompositeStationaryCudaGradient,
)

from benchmarks._cases import benchmark_cases
from benchmarks.pbe0_precision import reference

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires finite Slurm GPU allocation",
)
GRID = GridSpec(16, 6, 12)


@pytest.mark.parametrize(
    "case_name",
    (
        "water-def2-svp-spherical",
        "water-tetramer-def2-svp-spherical",
        "ammonia-def2-svp-spherical",
        "water-def2-svp",
    ),
)
def test_complete_candidate_against_independent_moving_grid(
    case_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep SCF, range exchange and VV10 unchanged; test cold/frozen/moved forces."""
    pytest.importorskip("pyscf")
    case = benchmark_cases()[case_name]
    moved = [(symbol, np.asarray(position).copy()) for symbol, position in case.atoms]
    moved[1][1][0] += 0.025
    coordinates = np.array([position for _, position in moved])
    oracles = (
        reference(case_name, GRID, 4, functional="WB97M_V"),
        reference(case_name, GRID, 4, functional="WB97M_V", atoms=moved),
    )
    selected = False
    records = []
    original = PreparedCompositeStationaryCudaGradient.execute

    def execute(owner: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(owner, *args, fp32_semilocal_geometry=selected, **kwargs)
        records.append(result[1])
        return result

    monkeypatch.setattr(PreparedCompositeStationaryCudaGradient, "execute", execute)
    calculator = Calculator(
        method="wb97m-v-rks",
        basis=case.generativeqc_basis,
        basis_representation=case.basis_representation,
        device="cuda",
        precision="fp64",
        ks_options=KsOptions(grid=GRID),
        initial_guess=None,
        max_iterations=180,
        energy_tolerance=1e-10,
        density_tolerance=1e-8,
        screening_tolerance=1e-12,
    )
    try:
        with calculator.prepare_batch([case.atoms], warm_start=True) as batch:
            for geometry, oracle in zip((None, [coordinates]), oracles, strict=True):
                selected = False
                batch.set_warm_start_updates(True)
                strict = batch.execute(geometry, strict=True).items[0]
                assert abs(strict.energy - oracle["energy"]) < 1e-7
                np.testing.assert_allclose(
                    strict.forces, oracle["forces"], atol=1e-6, rtol=0
                )
                assert (
                    records[-1]["stationary_source_work"]["semilocal"][
                        "fp32_geometry_points"
                    ]
                    == 0
                )
                selected = True
                batch.set_warm_start_updates(False)
                for _ in range(2):
                    mixed = batch.execute(geometry, strict=True).items[0]
                    assert abs(mixed.energy - oracle["energy"]) < 1e-7
                    np.testing.assert_allclose(
                        mixed.forces, oracle["forces"], atol=1e-5, rtol=0
                    )
                    assert np.sqrt(np.mean((mixed.forces - strict.forces) ** 2)) < 3e-6
                    np.testing.assert_allclose(
                        mixed.forces.sum(axis=0), 0, atol=2e-8, rtol=0
                    )
                    work = records[-1]
                    assert work["fp32_semilocal_geometry_requested"]
                    assert (
                        work["stationary_source_work"]["semilocal"][
                            "fp32_geometry_points"
                        ]
                        == work["grid_points"]
                    )
                    assert (
                        work["stationary_source_work"]["nonlocal"][
                            "fp32_geometry_points"
                        ]
                        == 0
                    )
                    assert work["ao_collocation_point_visits"] == work["grid_points"]
                    assert work["geometry_point_visits"] == 2 * work["grid_points"]
                selected = False
                replay = batch.execute(geometry, strict=True).items[0]
                np.testing.assert_allclose(
                    replay.forces, strict.forces, atol=1e-7, rtol=0
                )
    finally:
        calculator.clear_cache()


def test_numerical_retry_drains_pending_sources_and_isolates_failed_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inject after actual mixed launches; never combine rejected and strict sources."""
    from generativeqc._stationary_cuda import _CudaSources

    pytest.importorskip("pyscf")
    name = "water-def2-svp-spherical"
    case = benchmark_cases()[name]
    oracle = reference(name, GRID, 4, functional="WB97M_V")
    original_execute = PreparedCompositeStationaryCudaGradient.execute
    original_call = _CudaSources._call
    injected = False
    records = []

    def call(owner: Any, symbol: str, *args: Any) -> Any:
        nonlocal injected
        if (
            owner.fp32_geometry_requested
            and symbol == "stationary_finish_span"
            and not injected
        ):
            injected = True
            raise ArithmeticError("injected after enqueued FP32 geometry")
        return original_call(owner, symbol, *args)

    def execute(owner: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_execute(owner, *args, fp32_semilocal_geometry=True, **kwargs)
        records.append(result[1])
        return result

    monkeypatch.setattr(_CudaSources, "_call", call)
    monkeypatch.setattr(PreparedCompositeStationaryCudaGradient, "execute", execute)
    calculator = Calculator(
        method="wb97m-v-rks",
        basis=case.generativeqc_basis,
        basis_representation=case.basis_representation,
        device="cuda",
        precision="fp64",
        ks_options=KsOptions(grid=GRID),
        initial_guess=None,
        max_iterations=180,
        energy_tolerance=1e-10,
        density_tolerance=1e-8,
        screening_tolerance=1e-12,
    )
    try:
        with calculator.prepare_batch([case.atoms, case.atoms]) as batch:
            result = batch.execute([[0.0], None], strict=False)
            assert result.failure_indices == (0,)
            assert result.items[0].forces is None
            assert injected and len(records) == 1
            work = records[0]
            assert work["fp64_geometry_retries"] == 1
            assert work["rejected_semilocal_work"]["fp32_geometry_points"] > 0
            assert (
                work["stationary_source_work"]["semilocal"]["fp32_geometry_points"] == 0
            )
            np.testing.assert_allclose(
                result.items[1].forces, oracle["forces"], atol=1e-6, rtol=0
            )
            result = batch.execute(strict=True)
            for item, work in zip(result.items, records[-2:], strict=True):
                assert work["fp64_geometry_retries"] == 0
                assert (
                    work["stationary_source_work"]["semilocal"]["fp32_geometry_points"]
                    == work["grid_points"]
                )
                np.testing.assert_allclose(
                    item.forces, oracle["forces"], atol=1e-5, rtol=0
                )
    finally:
        calculator.clear_cache()
