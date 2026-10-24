"""Retirement guards for compiler-owned Direct source contraction."""

import subprocess
from pathlib import Path

from _cpp_source_support import cpp_function_definition
from conftest import NativeCxx
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


def test_materialized_recurrence_covers_all_spdf_shells(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Every Cartesian component fits in one whole-shell owner through ffff."""
    source = emit_direct_source_contraction_header()
    helpers = "\n".join(
        cpp_function_definition(source, name)
        for name in (
            "canonical_materialized_component_lanes",
            "canonical_materialized_component_slots",
        )
    )
    driver = tmp_path / "coverage.cpp"
    driver.write_text(
        "#include <cassert>\n#define __host__\n#define __device__\n"
        + helpers
        + r"""
int main() {
  for (unsigned first=0; first<=3; ++first)
  for (unsigned second=0; second<=3; ++second)
  for (unsigned third=0; third<=3; ++third)
  for (unsigned fourth=0; fourth<=3; ++fourth) {
    auto components=[](unsigned angular) { return (angular+1)*(angular+2)/2; };
    const auto order=first+second+third+fourth;
    const auto lanes=canonical_materialized_component_lanes(order);
    const auto slots=canonical_materialized_component_slots(order);
    const auto count=components(first)*components(second)*components(third)*components(fourth);
    assert(lanes > 0 && lanes <= 1024 && lanes%32 == 0);
    assert(slots*lanes >= count);
    for (unsigned component=0; component<count; ++component)
      assert(component/lanes < slots);
  }
}
"""
    )
    executable = tmp_path / "coverage"
    native_cxx.build_executable([driver], executable, compile_args=["-std=c++20"])
    subprocess.run([str(executable)], check=True, timeout=15)


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
    preparation = (ROOT / "src/scf/cuda/direct_jk.cpp").read_text(encoding="utf-8")
    assert "direct_jk_product(order.size(), 7 * sizeof(std::uint64_t))" in preparation
    assert "angular_order = 0; angular_order <= 12" in preparation


def test_axis_owners_borrow_only_published_geometry() -> None:
    """Parallel axes preserve full initialization and the legacy serial consumer."""
    source = emit_direct_source_contraction_header()
    assert "unsigned axis_begin = 0U, unsigned axis_end = 3U" in source
    assert "for (unsigned axis = axis_begin; axis < axis_end; ++axis)" in source
    assert "prepare_materialized_direct_pair(batch.shell_primitive_pairs[bra]" in source
    assert "prepare_materialized_direct_pair(batch.shell_primitive_pairs[ket]" in source
    assert "shared.bra, threadIdx.x, threadIdx.x + 1U)" in source
    assert "shared.ket, threadIdx.x, threadIdx.x + 1U)" in source
    begin = source.index(
        "__device__ inline void contract_materialized_direct_pair_fock("
    )
    end = source.index("}  // namespace generativeqc::scf::cuda_execution", begin)
    assert (
        source[begin:end].count("if (work) atomicAdd(&work->bra_preparations, 1ULL)")
        == 1
    )
