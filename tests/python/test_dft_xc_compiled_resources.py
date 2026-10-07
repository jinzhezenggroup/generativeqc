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
            "generativeqc::dft::cuda_xc_detail::(anonymous namespace)::ao_radial_kernel_4(double*)",
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
            "generativeqc::dft::cuda_xc_detail::evaluate_points<4, false, false>(double*)",
            registers=80,
            spill_store_bytes=16 if spill else 0,
            spill_load_bytes=8 if spill else 0,
        ),
        resource(
            "generativeqc::dft::cuda_xc_detail::evaluate_points<4, false, true>(double*)",
            registers=80,
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
            "generativeqc::dft::cuda_xc_detail::evaluate_points<5, false, false>(double*)",
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
    shape = GridXcCompiledResourceShape(
        ao_radial_reuse=True, npoint=4096, tile_points=256, nao=96, spins=2
    )
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


@pytest.mark.parametrize("npoint", [256, 512])
@pytest.mark.parametrize("enabled", [False, True])
def test_automatic_point_resource_envelope_retains_fallback(
    npoint: int, enabled: bool
) -> None:
    """The guarded default accounts batch pressure only when it is reachable."""
    rows = tuple(
        replace(row, registers=96, spill_store_bytes=32, spill_load_bytes=16)
        if "false, true>" in row.function
        else row
        for row in pbe_resources()
    )
    evidence = native_grid_xc_compiled_region_evidence(
        rows,
        shape=GridXcCompiledResourceShape(
            ao_radial_reuse=True,
            npoint=npoint,
            tile_points=256,
            nao=96,
            spins=2,
            point_batching=enabled,
        ),
        functional="PBE",
        target=TARGET,
        source_identity="guarded-point-default",
    )
    reachable = enabled and npoint > 256
    points = dict(evidence.scopes)["xc_points"]
    assert len(points) == (2 if reachable else 1)
    assert "false, false>" in points[0].function
    assert evidence.profitability.compiled_registers_per_thread == (
        96 if reachable else 80
    )
    assert evidence.profitability.spill_bytes == (48 if reachable else 0)
    with pytest.raises(ValueError, match="binding identity is stale"):
        replace(evidence, shape=replace(evidence.shape, point_batching=not enabled))
    with pytest.raises(TypeError, match="point-batching selector must be boolean"):
        replace(evidence.shape, point_batching=1)


def test_missing_reachable_batch_kernel_fails_closed() -> None:
    """Serial evidence cannot hide missing compiled coverage of the new default."""
    rows = tuple(row for row in pbe_resources() if "false, true>" not in row.function)
    with pytest.raises(ValueError, match="missing functional-specific batched XC"):
        native_grid_xc_compiled_region_evidence(
            rows,
            shape=GridXcCompiledResourceShape(
                ao_radial_reuse=True, npoint=4096, tile_points=256, nao=96, spins=2
            ),
            functional="PBE",
            target=TARGET,
            source_identity="missing-default-batch",
        )


@pytest.mark.parametrize(
    "missing",
    (
        "ao_radial_kernel_4",
        "tiled_density_product",
        "density_features",
        "evaluate_points<4",
        "tiled_potential",
    ),
)
def test_complete_region_fails_closed_when_one_scope_is_missing(missing: str) -> None:
    shape = GridXcCompiledResourceShape(
        ao_radial_reuse=True, npoint=4096, tile_points=256, nao=96, spins=2
    )
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
    shape = GridXcCompiledResourceShape(
        ao_radial_reuse=True, npoint=32, tile_points=16, nao=7, spins=2
    )
    rows = (
        resource("validate_density(double*)"),
        resource("ao_radial_kernel_4(double*)"),
        resource("density_product<false>(double*)", registers=61),
        resource("density_features<false>(double*)", registers=62),
        resource("evaluate_points<4, false, false>(double*)", registers=63),
        resource("evaluate_points<4, false, true>(double*)", registers=63),
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
    shape = GridXcCompiledResourceShape(
        ao_radial_reuse=True, npoint=4100, tile_points=256, nao=96, spins=2
    )
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
    assert any(
        "density_product<false>" in row.function and "tiled_" not in row.function
        for row in density
    )
    assert any("tiled_potential" in row.function for row in potential)
    assert any("assemble_potential" in row.function for row in potential)
    assert evidence.profitability.compiled_registers_per_thread == 92


def test_compiled_evidence_cannot_be_relabelled_to_another_shape() -> None:
    shape = GridXcCompiledResourceShape(
        ao_radial_reuse=True, npoint=4096, tile_points=256, nao=96, spins=2
    )
    first = native_grid_xc_compiled_region_evidence(
        pbe_resources(),
        shape=shape,
        functional="PBE",
        target=TARGET,
        source_identity="s" * 64,
    )
    with pytest.raises(ValueError, match="binding identity is stale"):
        replace(first, shape=replace(shape, npoint=8192))


@pytest.mark.parametrize("suffix", ["", "l", "ll"])
@pytest.mark.parametrize(
    ("functional", "feature_terms", "registers", "spill_bytes"),
    [("LDA_XC_PW", 1, 72, 0), ("PBE", 4, 192, 96)],
)
def test_point_specialization_tracks_feature_width_not_functional_code(
    suffix: str, functional: str, feature_terms: int, registers: int, spill_bytes: int
) -> None:
    shape = GridXcCompiledResourceShape(
        ao_radial_reuse=True,
        npoint=4096,
        tile_points=256,
        nao=96,
        spins=2,
        point_batching=False,
    )
    rows = (
        *(row for row in pbe_resources() if "evaluate_points" not in row.function),
        resource("ao_kernel(double*)", registers=52),
        resource(f"evaluate_points<1{suffix}, false, false>(double*)", registers=32),
        resource(
            f"evaluate_points<4{suffix}, false, false>(double*)",
            registers=192,
            spill_store_bytes=64,
            spill_load_bytes=32,
        ),
        resource(f"evaluate_points<5{suffix}, false, false>(double*)", registers=255),
        resource(f"evaluate_points<4{suffix}, true, false>(double*)", registers=254),
        resource(f"evaluate_points<40{suffix}, false, false>(double*)", registers=253),
        resource(f"evaluate_points<1{suffix}, false, true>(double*)", registers=252),
        resource(f"evaluate_points<4{suffix}, false, true>(double*)", registers=251),
    )
    evidence = native_grid_xc_compiled_region_evidence(
        rows,
        shape=shape,
        functional=functional,
        target=TARGET,
        source_identity="native-feature-width",
    )
    assert tuple(row.function for row in dict(evidence.scopes)["xc_points"]) == (
        f"evaluate_points<{feature_terms}{suffix}, false, false>(double*)",
    )
    assert evidence.profitability.compiled_registers_per_thread == registers
    assert evidence.profitability.spill_bytes == spill_bytes


@pytest.mark.parametrize(
    "inactive",
    [
        "1, false, false",
        "4, true, false",
        "40, false, false",
        "5, false, false",
        "4, false, true",
        "4, false",
    ],
)
def test_missing_pbe_width_cannot_be_replaced_by_an_inactive_kernel(
    inactive: str,
) -> None:
    rows = (
        *(row for row in pbe_resources() if "evaluate_points" not in row.function),
        resource(f"evaluate_points<{inactive}>(double*)"),
    )
    with pytest.raises(ValueError, match="functional-specific XC point kernel"):
        native_grid_xc_compiled_region_evidence(
            rows,
            shape=GridXcCompiledResourceShape(
                ao_radial_reuse=True, npoint=4096, tile_points=256, nao=96, spins=2
            ),
            functional="PBE",
            target=TARGET,
            source_identity="missing-native-pbe",
        )


@pytest.mark.parametrize(
    "inactive",
    [
        "ao_kernel",
        "ao_kernel_fp32",
        "ao_radial_kernel_10",
        "ao_radial_kernel_4_fp32",
        "ao_radial_kernel_40",
    ],
)
def test_missing_pbe_radial_variant_fails_closed(inactive: str) -> None:
    rows = tuple(
        row for row in pbe_resources() if "ao_radial_kernel_4(" not in row.function
    )
    with pytest.raises(ValueError, match="strict-FP64 AO kernel"):
        native_grid_xc_compiled_region_evidence(
            (*rows, resource(f"{inactive}(double*)", registers=255)),
            shape=GridXcCompiledResourceShape(
                ao_radial_reuse=True, npoint=256, tile_points=256, nao=96, spins=2
            ),
            functional="PBE",
            target=TARGET,
            source_identity="missing-active-ao",
        )


def test_ao_resource_pressure_excludes_inactive_radial_variants() -> None:
    rows = (
        *pbe_resources(),
        resource("ao_kernel(double*)", registers=255),
        resource("ao_radial_kernel_10(double*)", registers=255, spill_store_bytes=256),
        resource("ao_radial_kernel_4_fp32(double*)", registers=255),
        resource("ao_radial_kernel_40(double*)", registers=255),
    )
    evidence = native_grid_xc_compiled_region_evidence(
        rows,
        shape=GridXcCompiledResourceShape(
            ao_radial_reuse=True, npoint=256, tile_points=256, nao=96, spins=2
        ),
        functional="PBE",
        target=TARGET,
        source_identity="active-ao-only",
    )
    assert evidence.profitability.compiled_registers_per_thread == 80
    assert evidence.profitability.spill_bytes == 0
    assert [row.function for row in dict(evidence.scopes)["ao_jets"]] == [
        rows[0].function,
        rows[1].function,
    ]


def test_default_ao_evidence_remains_scalar_and_selector_is_bound() -> None:
    rows = (*pbe_resources(), resource("ao_kernel(double*)", registers=53))
    shape = GridXcCompiledResourceShape(npoint=256, tile_points=256, nao=96, spins=2)
    evidence = native_grid_xc_compiled_region_evidence(
        rows,
        shape=shape,
        functional="PBE",
        target=TARGET,
        source_identity="default-scalar-ao",
    )
    assert not shape.ao_radial_reuse
    assert (
        tuple(row.function for row in dict(evidence.scopes)["ao_jets"])[-1]
        == "ao_kernel(double*)"
    )
    with pytest.raises(ValueError, match="binding identity is stale"):
        replace(evidence, shape=replace(shape, ao_radial_reuse=True))
    with pytest.raises(TypeError, match="selector must be boolean"):
        replace(shape, ao_radial_reuse=1)
    with pytest.raises(ValueError, match="strict-FP64 AO kernel"):
        native_grid_xc_compiled_region_evidence(
            pbe_resources(),
            shape=shape,
            functional="PBE",
            target=TARGET,
            source_identity="default-missing-scalar-ao",
        )
