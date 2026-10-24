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


def test_partition_precheck_keeps_the_exact_scientific_gate_and_generic_route(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    """Compile actual admission: nonowners never screen, owners retain its result."""
    source = (ROOT / "src/scf/cuda/direct_bounded_fallback.cu").read_text()
    helper = _definition(source, "template <int AngularOrder>")
    kernel_begin = source.index("void bounded_direct_shell_quartet_kernel(")
    begin = source.index("const std::size_t first_pair = max(", kernel_begin)
    end = source.index("if (survives)", begin)
    probe = tmp_path / "admission.cpp"
    probe.write_text(
        PREFIX + helper + ADMISSION_PREFIX + source[begin:end] + ADMISSION_DRIVER
    )
    executable = native_cxx.build_executable(
        [probe], tmp_path / "admission", compile_args=["-std=c++20"]
    )
    subprocess.run([executable], check=True, timeout=30)


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
        const bool complement = bounded_direct_angular_owner<-3>(batch, first_pair, second_pair);
        assert(complement == (!passes[7] && !passes[8]));
        assert(unsigned(complement) + unsigned(passes[7]) + unsigned(passes[8]) == 1U);
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


ADMISSION_PREFIX = r"""
#include <algorithm>
#include <initializer_list>
using std::max; using std::min;
struct Work { unsigned long long candidates{}, angular_rejected{}, enumerated{}, shell_screened{}; } work;
struct ActiveShellQuartetTile { unsigned first,second,tile; };
enum class DirectScreeningPurpose { Force };
unsigned screened{};
bool accepted{};
template<class T> T atomicAdd(T* target,T value) { const auto old=*target; *target+=value; return old; }
Work* direct_force_task_work(const DeviceBatch&,ActiveShellQuartetTile task) {
  assert(task.first==1 && task.second==0 && task.tile==0); return &work;
}
template<bool,DirectScreeningPurpose>
bool direct_shell_quartet_survives_screening(const DeviceBatch&,std::size_t first,std::size_t second,
 double threshold,const void* bounds,const void* density,const void* optional,bool exchange,bool coulomb) {
  assert(first==1 && second==0 && threshold==1e-14);
  assert(bounds==nullptr && density==nullptr && optional==nullptr && !exchange && coulomb);
  ++screened; return accepted;
}
template<int FixedAngularOrder> bool admit(const DeviceBatch& batch) {
  constexpr bool Force=true, Unrestricted=false;
  constexpr auto Purpose=DirectScreeningPurpose::Force;
  const unsigned first_candidate=0, second_candidate=1;
  auto* traversal=&work;
  const double screening_tolerance=1e-14;
  const void *shell_pair_bounds=nullptr,*shell_pair_density_bounds=nullptr;
  const bool exchange_only=false,coulomb_only=true;
"""

ADMISSION_DRIVER = r"""
  return survives;
}
template<int Order> void exercise() {
  const std::int32_t first[]{0,2},second[]{1,3};
  std::uint8_t angular[4]{};
  const DeviceBatch batch{angular,first,second};
  for(unsigned assignment=0;assignment<256;++assignment) {
    unsigned sum=0;
    for(unsigned shell=0;shell<4;++shell) { angular[shell]=(assignment>>(2*shell))&3U; sum+=angular[shell]; }
    const bool owns=Order==-1 || (Order==-2 ? sum!=8 : Order==-3 ? sum!=7 && sum!=8 : sum==Order);
    for(bool pass:{false,true}) {
      work={}; screened=0; accepted=pass;
      assert(admit<Order>(batch)==(owns && pass));
      assert(screened==unsigned(owns));
      assert(work.candidates==1 && work.angular_rejected==unsigned(!owns));
      assert(work.enumerated==unsigned(owns) && work.shell_screened==unsigned(owns && !pass));
    }
  }
}
int main(){exercise<-1>();exercise<-2>();exercise<-3>();exercise<6>();exercise<7>();exercise<8>();}
"""
