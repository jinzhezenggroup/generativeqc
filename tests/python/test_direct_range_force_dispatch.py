"""Execute the actual range-force host selector without requiring a GPU.

Launch seams record the selected contract and forwarded scientific inputs.
Real-device range force tests separately qualify the retained recurrence.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from _cpp_source_support import cpp_function_definition

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_range_force_preserves_domain_and_unproved_basis_fallback(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Proved s/p/d LR calls and generic SR/f calls preserve every launch input."""
    source = (ROOT / "src/scf/cuda/direct_bounded_fallback.cu").read_text()
    dispatch = cpp_function_definition(
        source, "launch_bounded_direct_range_exchange_force_kernel"
    )
    # Only CUDA's launch punctuation is replaced; compile the production
    # conditions, template arguments and argument forwarding as ordinary C++.
    dispatch = re.sub(r"<<<[^>]*>>>", "", dispatch)
    probe = tmp_path / "range-force-dispatch.cpp"
    probe.write_text(_HEADER + dispatch + _MAIN)
    binary = native_cxx.build_executable(
        [probe], tmp_path / "probe", compile_args=("-std=c++20", "-O2")
    )
    subprocess.run([str(binary)], check=True, timeout=30)


_HEADER = r"""
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <tuple>
using cudaStream_t = void*;
struct dim3 { unsigned x, y=1, z=1; };
struct DeviceBatch { unsigned direct_maximum_shell_angular; };
struct ShellPairDensityBounds {};
enum class DirectScreeningPurpose { Fock, Force };
enum class DirectRangeOperator { Full, Short, Long };
namespace detail { struct BoundedDirectBlockDomain { unsigned cookie; }; }
constexpr unsigned kBoundedDirectThreads = 256, kBoundedDirectForceThreads = 128;
unsigned calls, maximum;
bool observed_spin;
double expected_omega, expected_coefficient;
DirectRangeOperator expected_range;
double output[1], density[1], bounds[1], block_bounds[1], system_bounds[1], schwarz[1];
ShellPairDensityBounds pair_density[1];
std::uint32_t order[1], overflow[1];
std::uint8_t active[1];
unsigned long long cursor;
template<bool U, DirectScreeningPurpose Purpose, bool Force,
         int Order=-1, int Radial=-1, bool Pair=false, bool Coop=false,
         bool Pure=false, unsigned Maximum=12, class... Args>
void bounded_direct_shell_quartet_kernel(Args... args) {
  const auto v=std::tuple{args...};
  static_assert(Purpose==DirectScreeningPurpose::Force && Force);
  assert(std::get<1>(v)==1e-12 && std::get<2>(v)==bounds);
  assert(std::get<3>(v)==pair_density && std::get<4>(v)==order);
  assert(std::get<5>(v)==block_bounds && std::get<6>(v)==system_bounds);
  assert(std::get<7>(v)==nullptr && std::get<8>(v)==0);
  assert(std::get<9>(v)==overflow && std::get<10>(v)==schwarz);
  assert(std::get<11>(v)==density && std::get<12>(v)==active);
  assert(std::get<13>(v)==output && std::get<14>(v)==&cursor);
  assert(std::get<15>(v)==nullptr && std::get<16>(v)==0);
  assert(std::get<17>(v)==expected_coefficient);
  assert(std::get<18>(v)==expected_range && std::get<19>(v)==expected_omega);
  assert(std::get<20>(v)==0 && !std::get<21>(v) && std::get<22>(v));
  assert(std::get<23>(v).cookie==37);
  if constexpr(Maximum<12) {
    assert(std::get<0>(v).direct_maximum_shell_angular<=2);
    assert(expected_range==DirectRangeOperator::Long);
    assert(Radial==static_cast<int>(expected_range));
  }
  ++calls; maximum=Maximum; observed_spin=U;
}
"""

_MAIN = r"""
int main() {
  for(bool spin: {false,true})
    for(unsigned angular: {0U,1U,2U,3U,255U})
      for(auto range: {DirectRangeOperator::Full,DirectRangeOperator::Short,
                       DirectRangeOperator::Long})
        for(double omega: {0.3,0.300001,0.0})
          for(auto block: {dim3{256},dim3{128},dim3{64,2}}) {
            calls=0; expected_range=range; expected_omega=omega;
            expected_coefficient=spin ? -0.46 : -0.23;
            launch_bounded_direct_range_exchange_force_kernel(
                spin,{4},block,8192,nullptr,{angular},1e-12,bounds,pair_density,
                order,block_bounds,system_bounds,overflow,schwarz,density,active,
                output,&cursor,range,omega,expected_coefficient,{37});
            if(range==DirectRangeOperator::Full) { assert(calls==0); continue; }
            assert(calls==1 && observed_spin==spin);
            assert(maximum==(angular<=2 && range==DirectRangeOperator::Long ? 8U:12U));
          }
}
"""
