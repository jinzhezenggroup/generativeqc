"""Guard optional pair-derivative dispatch independently of screening purpose."""

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_pair_derivative_selector_preserves_spin_screening_and_force_owner() -> None:
    """Both optional specializations write forces for either screening purpose."""
    source = (ROOT / "src/scf/cuda/direct_bounded_fallback.cu").read_text()
    begin = source.index("void launch_bounded_direct_shell_quartet_kernel_scaled(")
    end = source.index("void launch_bounded_direct_fock_shell_quartet_kernel(", begin)
    dispatch = source[begin:end]
    assert (
        "bounded_direct_shell_quartet_kernel<Unrestricted, Purpose, true, -1, -1, PairDerivatives>"
        in dispatch
    )
    for spin in ("true", "false"):
        for purpose in ("Fock", "Force"):
            assert (
                f"select.template operator()<{spin}, DirectScreeningPurpose::{purpose}>();"
                in dispatch
            )
    assert (
        "separate_sources && materialized_pair_derivative_available(batch)" in dispatch
    )
    for enabled in ("true", "false"):
        assert (
            f"launch.template operator()<Unrestricted, Purpose, {enabled}>();"
            in dispatch
        )


def test_full_range_force_promotes_qualified_static_128_thread_schedule() -> None:
    """Keep #1978's measured CTA width as the generic full-range force default."""
    source = (ROOT / "src/scf/cuda/direct_bounded_fallback.cu").read_text()
    constants = (ROOT / "src/scf/cuda/direct_constants.hpp").read_text()
    begin = source.index("void launch_bounded_direct_shell_quartet_kernel_scaled(")
    end = source.index("void launch_bounded_direct_range_exchange_force_kernel(", begin)
    dispatch = source[begin:end]

    assert "constexpr unsigned kBoundedDirectForceThreads = 128;" in constants
    assert "Force ? blockDim.x : detail::kBoundedDirectQueueCapacity" in source
    assert "slot += blockDim.x / detail::kDirectQuartetThreads" in source
    assert "block.x == kBoundedDirectThreads" in dispatch
    assert "block.x = kBoundedDirectForceThreads;" in dispatch
    assert "GENERATIVEQC_EXPERIMENT_DIRECT_FORCE_CTA_THREADS" not in source


def test_force_launch_width_preserves_materialized_component_coverage(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    """Execute the production selector and cover both dddd AO domains on the host."""
    source = (ROOT / "src/scf/cuda/direct_bounded_fallback.cu").read_text()
    begin = source.index("void launch_bounded_direct_shell_quartet_kernel_scaled(")
    launch = source.index("  auto launch =", begin)
    body = source.index("{", launch) + 1
    end = source.index("    const auto workspace_bytes =", body)
    # A common pre-dispatch reduction would also shrink PairDerivatives=true.
    assert "block.x =" not in source[begin:launch]
    selector = source[body:end]
    assert "if constexpr (!PairDerivatives)" in selector
    generator = (
        ROOT
        / "python/generativeqc_compiler/integral/direct_pair_materialized_gradient_cuda.py"
    ).read_text()
    assert "constexpr unsigned Slots = 6;" in generator
    assert "slot * detail::kDirectQuartetTileSize + threadIdx.x" in generator
    probe = tmp_path / "force_launch_width.cpp"
    probe.write_text(
        """
#include <cassert>
#include <cstddef>
#include <vector>
struct dim3 { unsigned x, y = 1, z = 1; };
constexpr unsigned kBoundedDirectThreads = 256;
constexpr unsigned kBoundedDirectForceThreads = 128;
template <bool PairDerivatives> dim3 select(dim3 block) {
"""
        + selector
        + """
  return block;
}
int main() {
  assert(select<false>({256}).x == 128);
  assert(select<true>({256}).x == 256);
  for (unsigned width : {64U, 128U, 192U}) {
    assert(select<false>({width}).x == width);
    assert(select<true>({width}).x == width);
  }
  for (dim3 block : {dim3{256, 2, 1}, dim3{256, 1, 2}}) {
    const auto generic = select<false>(block), materialized = select<true>(block);
    assert(generic.x == 256 && generic.y == block.y && generic.z == block.z);
    assert(materialized.x == 256 && materialized.y == block.y && materialized.z == block.z);
  }
  // Same-pair triangular and distinct-pair rectangular Cartesian dddd domains.
  for (std::size_t count : {666U, 1296U}) {
    std::vector<unsigned> visits(count);
    for (unsigned slot = 0; slot < 6; ++slot)
      for (unsigned lane = 0; lane < select<true>({256}).x; ++lane) {
        const auto ordinal = slot * 256U + lane;
        if (ordinal < count) ++visits[ordinal];
      }
    for (unsigned visits_per_component : visits) assert(visits_per_component == 1);
  }
}
"""
    )
    executable = native_cxx.build_executable(
        [probe], tmp_path / "force_launch_width", compile_args=["-std=c++20"]
    )
    subprocess.run([str(executable)], check=True, timeout=30)
