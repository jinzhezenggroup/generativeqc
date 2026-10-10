"""Retirement guards for compiler-owned Direct source contraction."""

from pathlib import Path

from generativeqc_compiler.integral.direct_source_contraction_cuda import (
    emit_direct_source_contraction_header,
)

ROOT = Path(__file__).resolve().parents[2]


def test_direct_source_contraction_is_compiler_owned() -> None:
    source = emit_direct_source_contraction_header()
    assert source.startswith("#pragma once\n")
    assert (
        "Generated from the compiler-owned Direct source-contraction lowering."
        in source
    )
    assert "contracted_eri_cartesian_source_shell_class(" in source
    assert "dispatch_contracted_eri_cartesian_source_shell_class" in source
    assert "weight * primitive_eri_cartesian_shell_class" in source
    assert "direct_native_source_contraction.cuh" not in source


def test_cartesian_source_contraction_accepts_runtime_radial_identity() -> None:
    source = emit_direct_source_contraction_header()
    assert "generativeqc::integrals::CoulombRange range" in source
    assert "double omega = 0.0" in source
    assert "primitive_eri_cartesian_shell_pairs<FirstShellAngular" in source
    assert "angular_fourth, range, omega" in source
    assert "range == generativeqc::integrals::CoulombRange::Full" in source


def test_native_source_contraction_owner_is_retired() -> None:
    native = ROOT / "src/scf/cuda/direct_native_source_contraction.cuh"
    assert not native.exists()
    for relative in (
        "src/scf/cuda/direct_fock_quartet.cuh",
        "src/scf/cuda/direct_force_quartet.cuh",
        "src/scf/cuda/direct_schwarz_kernels.cu",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert '#include "generated_direct_source_contraction.cuh"' in source
        assert "direct_native_source_contraction.cuh" not in source


def test_order_two_value_shortcut_preserves_derivative_seeds() -> None:
    """The value-only vector must not zero the full-range Dual3 force jet."""
    source = emit_direct_source_contraction_header()
    assert "std::is_arithmetic_v<Scalar>" in source
    shortcut = source.index("if constexpr ((ShellClass == 2")
    assert source.index("std::is_arithmetic_v<Scalar>", shortcut) < source.index(
        "Order2IntegralVector integral", shortcut
    )
    fallback = source.index(
        "return contracted_eri_cartesian_source_shell_class<FirstShellAngular",
        shortcut,
    )
    assert "derivative_coordinate, range, omega" in source[fallback:]


def test_source_contraction_generation_is_registered() -> None:
    generated = (ROOT / "cmake/GenerativeQCGeneratedSources.cmake").read_text(
        encoding="utf-8"
    )
    cuda = (ROOT / "cmake/GenerativeQCCuda.cmake").read_text(encoding="utf-8")
    assert "GENERATIVEQC_DIRECT_SOURCE_CONTRACTION_HEADER" in generated
    assert "generate_direct_source_contraction.py" in generated
    assert cuda.count("GENERATIVEQC_DIRECT_SOURCE_CONTRACTION_HEADER") == 2


def test_materialized_canonical_sources_reuse_range_and_scatter_owners() -> None:
    """Separate canonical sources must not inherit the HF exchange sign."""
    source = emit_direct_source_contraction_header()
    assert "double* separate_coulomb = nullptr" in source
    assert "double* separate_exchange = nullptr" in source
    assert "fill_range_coulomb<AngularOrder>" in source
    assert "shared.second.product_center, range, omega, shared.coulomb" in source
    assert "value[slot], true, false" in source
    assert "value[slot], false, true" in source
    assert "range, omega, shared.coulomb" in source


def test_materialized_recurrence_retains_qualified_order_bounds() -> None:
    """Multi-packet canonical reuse retains the established scientific owner."""
    source = emit_direct_source_contraction_header()
    assert "AngularOrder >= 3 && AngularOrder <= kMaximumCoulombOrder" in source
    assert "sizeof(MaterializedDirectPairRecurrence<12>) <= (48U << 10)" in source


def test_small_canonical_ctas_preserve_packet_offsets_and_slot_coverage() -> None:
    """Smaller owners change slot stride, never the legacy 256-component tile."""
    source = emit_direct_source_contraction_header()
    assert "canonical_materialized_component_lanes" in source
    assert "canonical_materialized_component_slots" in source
    assert "unsigned ComponentLanes = detail::kDirectQuartetTileSize" in source
    assert source.count("slot * ComponentLanes + threadIdx.x") == 2
    assert source.count("std::size_t(task.tile) * detail::kDirectQuartetTileSize") == 2
    assert "__syncthreads_or(any_admitted)" in source
    native = (ROOT / "src/scf/cuda/direct_jk_kernels.cu").read_text(encoding="utf-8")
    assert "canonical_materialized_component_lanes(angular_order)" in native
    assert "canonical_materialized_component_slots(AngularOrder)" in native
    assert "std::min(first_count, maximum_blocks)" in native
    assert "std::min(second_count, maximum_blocks)" in native
    assert "std::min(dense_upper_bound, maximum_blocks)" in native
