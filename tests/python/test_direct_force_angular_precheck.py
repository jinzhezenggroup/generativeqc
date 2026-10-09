"""Exact angular ownership must precede, not replace, scientific screening."""

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from test_direct_force_schedule_composition import _definition

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_angular_owner_is_exact_for_all_spdf_physical_quartets(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    """Execute the actual helper over repeated shells and reordered physical pairs."""
    source = (ROOT / "src/scf/cuda/direct_bounded_fallback.cu").read_text()
    helper = _definition(source, "template <int AngularOrder>")
    probe = tmp_path / "angular_owner.cpp"
    probe.write_text(PREFIX + helper + DRIVER)
    executable = native_cxx.build_executable(
        [probe], tmp_path / "angular_owner", compile_args=["-std=c++20"]
    )
    subprocess.run([str(executable)], check=True, timeout=30)


def test_partition_precheck_keeps_the_exact_scientific_gate_and_generic_route() -> None:
    """No owning task bypasses screening; non-owning threads still reach CTA barriers."""
    source = (ROOT / "src/scf/cuda/direct_bounded_fallback.cu").read_text()
    kernel_begin = source.index("void bounded_direct_shell_quartet_kernel(")
    kernel = source[
        kernel_begin : source.index(
            "template <bool Unrestricted, DirectRangeOperator Range, unsigned Order = 0U>",
            kernel_begin,
        )
    ]
    admission = kernel[kernel.index("const std::size_t first_pair = max(") :]
    assert "bounded_direct_angular_owner<FixedAngularOrder>" in admission
    assert (
        "&&\n            direct_shell_quartet_survives_screening<Unrestricted, Purpose>"
        in admission
    )
    before_screen = admission[
        : admission.index("direct_shell_quartet_survives_screening")
    ]
    assert "continue" not in before_screen
    assert "candidate_order" not in admission
    assert "if (!generated_class)" in admission
    assert (
        "profile_bounded_direct_shell_quartet(batch, queue[slot], profile)" in admission
    )
    assert "__syncthreads();" in admission
    packing = (ROOT / "src/scf/cuda/direct_jk.cpp").read_text()
    assert (
        "*std::max_element(host.shell_angular.begin(), host.shell_angular.end())"
        in packing
    )
    basis = (ROOT / "src/scf/cuda/packed_basis.hpp").read_text()
    assert "direct_maximum_shell_angular{255U}" in basis


PREFIX = r"""
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#define __host__
#define __device__
struct DeviceBatch {
  const uint8_t* shell_angular{};
  const int32_t* shell_pair_first{};
  const int32_t* shell_pair_second{};
};
"""

DRIVER = r"""
template <std::size_t... Orders>
std::array<bool, sizeof...(Orders)> owning_passes(
    const DeviceBatch& batch, std::size_t first_pair, std::size_t second_pair,
    std::index_sequence<Orders...>) {
  return {bounded_direct_angular_owner<static_cast<int>(Orders)>(
      batch, first_pair, second_pair)...};
}
int main() {
  assert(bounded_direct_angular_owner<-1>({}, 100, 200));
  std::array<int32_t, 10> first_shells{}, second_shells{};
  std::size_t pair_count = 0;
  for (int32_t first_shell = 0; first_shell < 4; ++first_shell)
    for (int32_t second_shell = 0; second_shell <= first_shell; ++second_shell) {
      first_shells[pair_count] = first_shell;
      second_shells[pair_count++] = second_shell;
    }
  const std::array<std::size_t, 10> order{9, 0, 8, 1, 7, 2, 6, 3, 5, 4};
  std::array<uint8_t, 4> angular{};
  for (unsigned assignment = 0; assignment < 256; ++assignment) {
    for (unsigned shell = 0; shell < 4; ++shell)
      angular[shell] = (assignment >> (2 * shell)) & 3U;
    const DeviceBatch batch{angular.data(), first_shells.data(), second_shells.data()};
    for (std::size_t first_position = 0; first_position < pair_count; ++first_position)
      for (std::size_t second_position = 0; second_position <= first_position; ++second_position) {
        const auto first_pair = order[first_position], second_pair = order[second_position];
        const auto expected = angular[first_shells[first_pair]] + angular[second_shells[first_pair]] +
                              angular[first_shells[second_pair]] + angular[second_shells[second_pair]];
        const auto passes = owning_passes(batch, first_pair, second_pair,
                                         std::make_index_sequence<13>{});
        assert(bounded_direct_angular_owner<-2>(batch, first_pair, second_pair) == !passes[8]);
        unsigned owners = 0;
        for (std::size_t pass = 0; pass < passes.size(); ++pass) {
          assert(passes[pass] == (pass == static_cast<std::size_t>(expected)));
          owners += passes[pass];
        }
        assert(owners == 1);
        // Screening may accept/reject or see a NaN. Its invocation belongs to
        // exactly one pass in every case; ownership never decides acceptance.
        for (bool accepted : {false, true}) {
          unsigned scientific_calls = 0, admitted = 0;
          for (bool owns : passes)
            if (owns) { ++scientific_calls; admitted += accepted; }
          assert(scientific_calls == 1 && admitted == unsigned(accepted));
        }
      }
  }
}
"""
