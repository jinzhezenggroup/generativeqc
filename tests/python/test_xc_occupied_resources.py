"""Admit every compiled factor/fallback arm and exclude inactive FP32 facts."""

from dataclasses import replace

import pytest
from generativeqc_compiler.common.cuda_resources import KernelResources
from generativeqc_compiler.dft.xc_compiled_resources import (
    GridXcCompiledRegionEvidence,
    GridXcCompiledResourceShape,
    native_grid_xc_compiled_region_evidence,
)
from generativeqc_compiler.dft.xc_contraction_cuda import DEFAULT_XC_MATRIX_SCHEDULE
from test_dft_xc_compiled_resources import TARGET, pbe_resources, resource


def specialized_resources() -> tuple[KernelResources, ...]:
    incumbent = tuple(
        row
        for row in pbe_resources()
        if not row.function.startswith(
            ("batch_density_products", "batch_density_features")
        )
    )
    return incumbent + (
        resource("batch_density_products<double, false>(double*)", registers=34),
        resource("batch_density_products<double, true>(double*)", registers=96),
        resource("batch_density_features<false>(double*)", registers=36),
        resource("batch_density_features<true>(double*)", registers=48),
        resource("batch_density_products<float, false>(double*)", registers=255),
        resource("occupied_density_products(double*)", registers=112),
        resource("occupied_density_features(double*)", registers=48),
    )


def qualify(
    rows: tuple[KernelResources, ...], *, compact: bool = True
) -> GridXcCompiledRegionEvidence:
    return native_grid_xc_compiled_region_evidence(
        rows,
        shape=GridXcCompiledResourceShape(
            npoint=4096,
            tile_points=256,
            nao=96,
            spins=2,
            ao_radial_reuse=True,
            compact_batching=compact,
        ),
        functional="PBE",
        target=TARGET,
        source_identity="s" * 64,
    )


def test_specialized_resource_envelope_covers_factor_and_matrix_routes() -> None:
    result = qualify(specialized_resources())
    assert result.profitability.compiled_registers_per_thread == 112
    density = dict(result.scopes)["density_product"]
    assert any("double, false" in row.function for row in density)
    assert any("double, true" in row.function for row in density)
    assert any("occupied_density_products" in row.function for row in density)
    assert not any("float, false" in row.function for row in density)


def test_noncompact_projection_remains_in_the_resource_envelope() -> None:
    result = qualify(specialized_resources(), compact=False)
    assert result.profitability.compiled_registers_per_thread == 112
    density = dict(result.scopes)["density_product"]
    assert any("occupied_density_products" in row.function for row in density)
    assert not any("batch_density_products" in row.function for row in density)


def test_standalone_projection_occupancy_uses_its_full_block_width() -> None:
    rows = tuple(
        replace(
            row,
            registers=96 if row.function.startswith("occupied_density_products") else 8,
            shared_bytes=0,
        )
        for row in specialized_resources()
    )
    result = qualify(rows, compact=False)
    threads = DEFAULT_XC_MATRIX_SCHEDULE.threads
    resident_blocks = min(
        TARGET.maximum_blocks_per_sm,
        TARGET.maximum_threads_per_sm // threads,
        TARGET.registers_per_sm // (96 * threads),
    )
    assert result.profitability.compiled_occupancy_upper_bound == (
        resident_blocks * threads / TARGET.maximum_threads_per_sm
    )


@pytest.mark.parametrize(
    "missing",
    (
        "batch_density_products<double, false>",
        "batch_density_products<double, true>",
        "batch_density_features<false>",
        "batch_density_features<true>",
        "occupied_density_products",
        "occupied_density_features",
    ),
)
def test_missing_specialized_resource_fails_closed(missing: str) -> None:
    rows = tuple(
        row for row in specialized_resources() if not row.function.startswith(missing)
    )
    with pytest.raises(ValueError, match="specialized|standalone occupied"):
        qualify(rows)
