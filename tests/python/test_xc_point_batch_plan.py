"""Execute compiler-emitted residency admission without CUDA or runtime imports."""

import subprocess
from pathlib import Path

from generativeqc_compiler.dft.ao_cuda import emit_native_xc_contraction_kernels
from generativeqc_compiler.dft.xc_point_batch_cuda import (
    emit_native_xc_point_batch_plan,
)


def test_emitted_point_batch_resource_admission(
    tmp_path: Path, native_cxx: object
) -> None:
    """Ragged maps determine residency; insufficient budgets keep the incumbent."""
    source = tmp_path / "plan.cpp"
    source.write_text(
        "#include <algorithm>\n#include <climits>\n#include <cstddef>\n"
        "#include <limits>\n#include <stdexcept>\n#include <vector>\n"
        "enum class CudaXcAoPrecision { Fp64, Fp32ComputeFp64Storage };\n"
        "struct CudaXcLayout {\n"
        "  std::size_t npoint{103}, tile_points{16}, nao{64}, jets{4}, spins{2}, feature_terms{5};\n"
        "  bool local_ao{}, response{};\n"
        "  CudaXcAoPrecision ao_precision{CudaXcAoPrecision::Fp64};\n};\n"
        "struct CudaXcPointBatchPlan {\n"
        "  std::size_t tiles{1}, ao_elements{}, feature_elements{}, total_elements{}, device_bytes{};\n};\n"
        + emit_native_xc_point_batch_plan()
        + r"""
int main() {
  CudaXcLayout layout;
  const auto dense = prepare_point_batch_plan(layout, {}, 4, 1 << 20);
  if (dense.tiles != 4 || dense.ao_elements != 4 * 16 * 64 * 4 ||
      dense.feature_elements != 4 * 16 * 2 * 5 || dense.total_elements != 3 * 4 * 16 ||
      dense.device_bytes != (dense.ao_elements + 2 * dense.feature_elements + dense.total_elements) * 8)
    return 1;
  layout.local_ao = true;
  // Empty, high occupancy, nonuniform, and tail tiles; indexed AO residency
  // must not charge dense AO panels or collapse the original tile domains.
  const std::vector<std::size_t> offsets{0, 0, 64, 67, 68, 70, 70, 75};
  const auto mapped = prepare_point_batch_plan(layout, offsets, 4, dense.device_bytes);
  if (mapped.tiles != 4 || mapped.ao_elements != 16 * 68 * 4 ||
      mapped.device_bytes >= dense.device_bytes) return 2;
  const auto limited = prepare_point_batch_plan(layout, offsets, 4, mapped.device_bytes - 1);
  if (limited.tiles != 2 || limited.device_bytes >= mapped.device_bytes) return 3;
  for (const auto budget : {std::size_t{0}, std::size_t{1}, std::size_t{127}}) {
    const auto fallback = prepare_point_batch_plan(layout, offsets, 4, budget);
    if (fallback.tiles != 1 || fallback.device_bytes != 0) return 4;
  }
  if (prepare_point_batch_plan(layout, offsets, 1, dense.device_bytes).tiles != 1) return 5;
  layout.response = true;
  if (prepare_point_batch_plan(layout, offsets, 4, dense.device_bytes).tiles != 1) return 6;
  layout.response = false;
  layout.ao_precision = CudaXcAoPrecision::Fp32ComputeFp64Storage;
  if (prepare_point_batch_plan(layout, offsets, 4, dense.device_bytes).tiles != 1) return 7;
  layout.ao_precision = CudaXcAoPrecision::Fp64;
  const auto huge = prepare_point_batch_plan(layout, offsets,
                                             std::numeric_limits<std::size_t>::max(), 1 << 20);
  if (huge.tiles != 7 || huge.device_bytes > (1 << 20)) return 8;
  layout.npoint = 7;
  layout.local_ao = false;
  if (prepare_point_batch_plan(layout, {}, 16, 1 << 20).tiles != 1) return 9;
  layout.npoint = 103;
  layout.local_ao = true;
  for (const auto bad : {std::vector<std::size_t>{},
                         std::vector<std::size_t>{0, 0, 65, 65, 65, 65, 65, 65},
                         std::vector<std::size_t>{0, 2, 1, 1, 1, 1, 1, 1}}) {
    try { prepare_point_batch_plan(layout, bad, 4, 1 << 20); return 10; }
    catch (const std::invalid_argument&) {}
  }
}
""",
        encoding="utf-8",
    )
    binary = tmp_path / "plan"
    native_cxx.build_executable([source], binary, compile_args=("-std=c++17", "-O2"))
    subprocess.run([str(binary)], check=True)


def test_point_batches_share_canonical_consumer_and_launch_width() -> None:
    """Batching does not specialize/prune the canonical scientific consumer."""
    source = emit_native_xc_contraction_kernels()
    assert "evaluate_points<feature_terms, false, true>" in source
    assert "return &launch_point_batches<4, 32>;" in source
    assert "min(tile_points, domain_count - begin)" in source
    assert (
        source.count("xc = evaluate_semilocal_point(functional, rho, gradient, tau")
        == 1
    )
    assert "static_assert(!batched || !response)" in source
