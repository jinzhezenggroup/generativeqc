"""Complete compiled-resource evidence for native CUDA grid/XC regions."""

from __future__ import annotations

from dataclasses import replace

import pytest
from generativeqc_compiler.common.cuda_resources import KernelResources
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.dft.xc_compiled_resources import (
    GRID_XC_COMPILED_SCOPES,
    GridXcCompiledResourceShape,
    native_grid_xc_compiled_region_evidence,
)

TARGET = cuda_target_info("sm_120")


def resource(
    function: str,
    *,
    registers: int = 48,
    spill_store_bytes: int = 0,
    spill_load_bytes: int = 0,
    shared_bytes: int = 0,
    local_bytes: int | None = None,
) -> KernelResources:
    return KernelResources(
        function=function,
        registers=registers,
        stack_bytes=0,
        spill_store_bytes=spill_store_bytes,
        spill_load_bytes=spill_load_bytes,
        shared_bytes=shared_bytes,
        local_bytes=local_bytes,
    )


def pbe_resources(*, spill: bool = False) -> tuple[KernelResources, ...]:
    return (
        resource(
            "generativeqc::dft::cuda_xc_detail::(anonymous namespace)::validate_density(double*)",
            registers=24,
        ),
        resource(
            "generativeqc::dft::cuda_xc_detail::(anonymous namespace)::ao_kernel(double*)",
            registers=52,
        ),
        resource(
            "generativeqc::dft::cuda_xc_detail::tiled_density_product<false>(double*)",
            registers=64,
            shared_bytes=4352,
        ),
        resource(
            "generativeqc::dft::cuda_xc_detail::density_features<true>(double*)",
            registers=56,
        ),
        resource(
            "generativeqc::dft::cuda_xc_detail::evaluate_points<1, false>(double*)",
            registers=80,
            spill_store_bytes=16 if spill else 0,
            spill_load_bytes=8 if spill else 0,
        ),
        resource(
            "generativeqc::dft::cuda_xc_detail::compact_potential_panels(double*)",
            registers=40,
        ),
        resource(
            "generativeqc::dft::cuda_xc_detail::tiled_potential(double*)",
            registers=72,
            shared_bytes=8704,
        ),
        # Compiled but inactive variants must not contaminate the selected region.
        resource(
            "generativeqc::dft::cuda_xc_detail::evaluate_points<2, false>(double*)",
            registers=255,
            spill_store_bytes=128,
            spill_load_bytes=128,
        ),
        resource(
            "generativeqc::dft::cuda_xc_detail::ao_kernel_fp32(double*)",
            registers=200,
        ),
    )


def test_complete_region_selects_only_active_pbe_scopes() -> None:
    shape = GridXcCompiledResourceShape(npoint=4096, tile_points=256, nao=96, spins=2)
    evidence = native_grid_xc_compiled_region_evidence(
        pbe_resources(),
        shape=shape,
        functional="PBE",
        target=TARGET,
        source_identity="s" * 64,
        object_bytes=1234,
        compile_seconds=2.5,
    )
    assert tuple(name for name, _ in evidence.scopes) == GRID_XC_COMPILED_SCOPES
    assert evidence.profitability.compiled_registers_per_thread == 80
    assert evidence.profitability.spill_bytes == 0
    assert evidence.profitability.shared_bytes == 8704
    assert evidence.profitability.object_bytes == 1234
    assert evidence.profitability.compile_seconds == 2.5
    assert evidence.profitability.compiled_occupancy_upper_bound is not None
    assert evidence.profitability.compiled_occupancy_upper_bound > 0
    # The inactive r2SCAN and FP32 variants are intentionally much worse.
    assert evidence.profitability.compiled_registers_per_thread < 200


@pytest.mark.parametrize(
    "missing",
    (
        "ao_kernel",
        "tiled_density_product",
        "density_features",
        "evaluate_points<1",
        "tiled_potential",
    ),
)
def test_complete_region_fails_closed_when_one_scope_is_missing(missing: str) -> None:
    shape = GridXcCompiledResourceShape(npoint=4096, tile_points=256, nao=96, spins=2)
    rows = tuple(row for row in pbe_resources() if missing not in row.function)
    with pytest.raises(ValueError, match="missing"):
        native_grid_xc_compiled_region_evidence(
            rows,
            shape=shape,
            functional="PBE",
            target=TARGET,
            source_identity="s" * 64,
        )


def test_small_ao_shape_selects_scalar_density_and_vxc_variants() -> None:
    shape = GridXcCompiledResourceShape(npoint=32, tile_points=16, nao=7, spins=2)
    rows = (
        resource("validate_density(double*)"),
        resource("ao_kernel(double*)"),
        resource("density_product<false>(double*)", registers=61),
        resource("density_features<false>(double*)", registers=62),
        resource("evaluate_points<1, false>(double*)", registers=63),
        resource("assemble_potential(double*)", registers=64),
        resource("accumulate_totals(double*)", registers=8),
        resource("tiled_density_product<false>(double*)", registers=250),
        resource("density_features<true>(double*)", registers=250),
        resource("tiled_potential(double*)", registers=250),
    )
    evidence = native_grid_xc_compiled_region_evidence(
        rows,
        shape=shape,
        functional="PBE",
        target=TARGET,
        source_identity="small-source",
    )
    assert evidence.profitability.compiled_registers_per_thread == 64
    assert "tiled_potential" not in {
        row.function for _, scope in evidence.scopes for row in scope
    }


def test_partial_final_tile_includes_tiled_and_scalar_resource_paths() -> None:
    shape = GridXcCompiledResourceShape(npoint=4100, tile_points=256, nao=96, spins=2)
    rows = (
        *pbe_resources(),
        resource("density_product<false>(double*)", registers=91),
        resource("assemble_potential(double*)", registers=92),
        resource("accumulate_totals(double*)", registers=8),
    )
    evidence = native_grid_xc_compiled_region_evidence(
        rows,
        shape=shape,
        functional="PBE",
        target=TARGET,
        source_identity="partial-source",
    )
    density = dict(evidence.scopes)["density_product"]
    potential = dict(evidence.scopes)["vxc_contraction"]
    assert any("tiled_density_product" in row.function for row in density)
    assert any("density_product<false>" in row.function and "tiled_" not in row.function for row in density)
    assert any("tiled_potential" in row.function for row in potential)
    assert any("assemble_potential" in row.function for row in potential)
    assert evidence.profitability.compiled_registers_per_thread == 92


def test_compiled_evidence_identity_changes_with_execution_shape() -> None:
    shape = GridXcCompiledResourceShape(npoint=4096, tile_points=256, nao=96, spins=2)
    first = native_grid_xc_compiled_region_evidence(
        pbe_resources(),
        shape=shape,
        functional="PBE",
        target=TARGET,
        source_identity="s" * 64,
    )
    second = replace(first, shape=replace(shape, npoint=8192))
    assert first.identity != second.identity
