"""Compile the actual implicit AO addressing and launch admission without CUDA."""

import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from test_stationary_task_work_budget import _block

if TYPE_CHECKING:
    from conftest import NativeCxx


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "src/scf/cuda/one_electron_gradient_launch.cuh").read_text()


def test_implicit_pairs_preserve_triangular_batch_and_warp_ownership(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    """Check every lane and repair sqrt rounding at 64-bit ordinal boundaries."""
    addressing = _block(SOURCE, "__device__ std::size_t implicit_pair_first(")
    cooperative = _block(
        SOURCE, "__global__ void implicit_nucleus_cooperative_gradient("
    )
    source = tmp_path / "pairs.cpp"
    source.write_text(
        PREFIX + addressing + "\ntemplate <class Program>\n" + cooperative + PAIR_DRIVER
    )
    executable = native_cxx.build_executable(
        [source], tmp_path / "pairs", compile_args=["-std=c++17", "-O2"]
    )
    subprocess.run([str(executable)], check=True, timeout=15)


def test_implicit_launch_validation_and_sticky_failure(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    """Fake only CUDA submission, retaining the actual native admission code."""
    launch = _block(SOURCE, "cudaError_t launch_gradient_family(")
    launch = re.sub(
        r"\w+<Program>\s*<<<(?P<configuration>.*?)>>>\s*\((?P<arguments>.*?)\);",
        lambda match: f"submit({match['configuration']}, {match['arguments']});",
        launch,
        flags=re.DOTALL,
    )
    source = tmp_path / "launch.cpp"
    source.write_text(PREFIX + LAUNCH_PREFIX + launch + LAUNCH_DRIVER)
    executable = native_cxx.build_executable(
        [source], tmp_path / "launch", compile_args=["-std=c++17", "-O2"]
    )
    subprocess.run([str(executable)], check=True, timeout=10)


def test_bridges_do_not_pack_unused_cooperative_pair_lists() -> None:
    """Keep explicit shell controls and the prepared overlap-only Pulay mapping."""
    bridge = (ROOT / "src/scf/cuda/one_electron_gradient_bridge.cu").read_text()
    assert "if (schedule == 1)" in bridge
    assert "if (schedule == 0)" in bridge
    assert "if (schedule == 0 || schedule == 3)" not in bridge
    assert "host.pair_first.size()" in bridge
    assert "one_electron_derivative_mapping_requested() == 3" in bridge
    assert "1, 1.0, output, source->stream" in bridge


def test_prepared_mapping_uses_cooperative_default_and_preserves_rollback(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    """Compile the real selector and bridge admission, including an unset knob."""
    bridge = (ROOT / "src/scf/cuda/one_electron_gradient_bridge.cu").read_text()
    selection = re.search(r"const unsigned hcore_schedule =.*?;", bridge, re.DOTALL)
    assert selection is not None
    policy = _block(
        (ROOT / "src/scf/cuda/rhf_policy.cpp").read_text(),
        "unsigned one_electron_derivative_mapping_requested() noexcept",
    )
    source = tmp_path / "mapping.cpp"
    source.write_text(
        "#include <cassert>\n#include <cstdlib>\n#include <cstring>\n#include <initializer_list>\n"
        "namespace cuda_policy {\n"
        "struct NucleusCooperativeSchedule { static constexpr unsigned schedule_code=3; };\n"
        + policy
        + "\n}\nunsigned prepared_mapping() {\n"
        + selection.group()
        + "\nreturn hcore_schedule;\n}\n"
        + r"""
int main() {
  const char* variable="GENERATIVEQC_ONE_ELECTRON_DERIVATIVE_MAPPING";
  assert(unsetenv(variable)==0);
  assert(cuda_policy::one_electron_derivative_mapping_requested()==3);
  assert(prepared_mapping()==3);
  for (const char* selection : {"nucleus_cooperative", "cooperative", "3"}) {
    assert(setenv(variable,selection,1)==0);
    assert(prepared_mapping()==3);
  }
  for (const char* selection : {"shell_warp", "1", "serial", "2", "0", "invalid", ""}) {
    assert(setenv(variable,selection,1)==0);
    assert(prepared_mapping()==1);
  }
}
"""
    )
    executable = native_cxx.build_executable(
        [source], tmp_path / "mapping", compile_args=["-std=c++17", "-O2"]
    )
    subprocess.run([str(executable)], check=True, timeout=10)


PREFIX = r"""
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>
#define __global__
#define __shared__
#define __device__
constexpr unsigned kCooperativeLanes=32,kCooperativeGroups=4,kWarpThreads=32;
namespace generated { struct PairGeometry {}; }
struct TestProgram { using Geometry=generated::PairGeometry; };
struct Dimension { std::size_t x; } blockIdx,blockDim,threadIdx;
struct OneElectronDeviceView {
  std::int32_t batch_size=1,nbf=1;
  std::size_t shell_pair_count=1;
  const std::int32_t *shell_pair_first{},*shell_pair_second{},*ao_shells{};
  const std::int64_t *shell_ao_offsets{},*atom_offsets{};
};
struct OneElectronWeightView {
  double overlap_scale=1,kinetic_scale=1,attraction_scale=1;
};
std::vector<std::pair<std::int64_t,std::int64_t>> visits;
template <class Program>
void contract_pair_nucleus_cooperative(OneElectronDeviceView,std::int64_t first,
    std::int64_t second,OneElectronWeightView,double sign,double*,
    typename Program::Geometry*) {
  assert(sign==-1);
  visits.emplace_back(first,second);
}
"""


PAIR_DRIVER = r"""
int main() {
  blockDim.x=kCooperativeLanes*kCooperativeGroups;
  for(int systems: {1,3}) for(bool mask: {false,true})
    for(int nbf: {1,2,3,8,29,32,64,127}) {
      OneElectronDeviceView batch{};
      batch.batch_size=systems;
      batch.nbf=nbf;
      const std::size_t count=std::size_t(nbf)*(nbf+1)/2;
      std::uint8_t active[]{1,0,1};
      std::map<std::pair<std::int64_t,std::int64_t>,int> counts;
      for(std::size_t task=0;task<systems*count+4;++task) {
        blockIdx.x=task/kCooperativeGroups;
        std::vector<std::pair<std::int64_t,std::int64_t>> ordered;
        for(unsigned lane=0;lane<kCooperativeLanes;++lane) {
          threadIdx.x=task%kCooperativeGroups*kCooperativeLanes+lane;
          visits.clear();
          implicit_nucleus_cooperative_gradient<TestProgram>(
              batch,count,{},mask?active:nullptr,-1,nullptr);
          if(lane==0) {
            ordered=visits;
            for(auto pair:visits) ++counts[pair];
          } else assert(visits==ordered);
        }
      }
      for(int system=0;system<systems;++system)
        for(int first=0;first<nbf;++first) for(int second=0;second<nbf;++second) {
          const auto pair=std::make_pair(system*nbf+first,system*nbf+second);
          assert(counts[pair]==int(first>=second && (!mask || active[system])));
        }
      for(const auto& entry:counts) assert(entry.first.first/nbf==entry.first.second/nbf);
    }
  for(std::size_t nbf: {1U,3U,32U,65535U,2147483647U})
    for(std::size_t row: {std::size_t(0),nbf/2,nbf-1}) {
      const auto begin=row*(row+1)/2;
      assert(implicit_pair_first(begin,nbf)==row);
      assert(implicit_pair_first(begin+row,nbf)==row);
      if(row) assert(implicit_pair_first(begin-1,nbf)==row-1);
      if(row+1<nbf) assert(implicit_pair_first(begin+row+1,nbf)==row+1);
    }
}
"""


LAUNCH_PREFIX = r"""
using cudaError_t=int;
using cudaStream_t=int;
constexpr int cudaSuccess=0,cudaErrorInvalidValue=1;
constexpr unsigned kDerivativeThreads=128;
struct NucleusCooperativeSchedule {
  static constexpr unsigned schedule_code=3,block_threads=128,groups_per_block=4;
};
enum class PairDomain { implicit_pairs, explicit_pairs, system_or_shell };
std::vector<PairDomain> kernels;
int sticky_error,fail_launch;
cudaError_t cudaPeekAtLastError() { return sticky_error; }
void record_submission(unsigned blocks,unsigned threads,unsigned shared,
                       cudaStream_t stream,PairDomain domain) {
  assert(blocks && threads==128 && shared==0 && stream==0);
  kernels.push_back(domain);
  if(int(kernels.size())==fail_launch) sticky_error=7;
}
void submit(unsigned blocks,unsigned threads,unsigned shared,cudaStream_t stream,
            OneElectronDeviceView,OneElectronWeightView,const std::uint8_t*,
            double,double*) {
  record_submission(blocks,threads,shared,stream,PairDomain::system_or_shell);
}
void submit(unsigned blocks,unsigned threads,unsigned shared,cudaStream_t stream,
            OneElectronDeviceView batch,std::size_t count,OneElectronWeightView,
            const std::uint8_t*,double,double*) {
  const auto n=std::size_t(batch.nbf);
  assert(count==n*(n+1)/2);
  record_submission(blocks,threads,shared,stream,PairDomain::implicit_pairs);
}
void submit(unsigned blocks,unsigned threads,unsigned shared,cudaStream_t stream,
            OneElectronDeviceView,const std::int32_t* first,const std::int32_t* second,
            std::size_t count,OneElectronWeightView,const std::uint8_t*,double,double*) {
  assert(first && second && count);
  record_submission(blocks,threads,shared,stream,PairDomain::explicit_pairs);
}
"""


LAUNCH_DRIVER = r"""
int main() {
  std::int32_t indices[]{0};
  std::int64_t offsets[]{0,1};
  double output{};
  OneElectronDeviceView batch{};
  batch.shell_pair_first=batch.shell_pair_second=batch.ao_shells=indices;
  batch.atom_offsets=batch.shell_ao_offsets=offsets;
  const auto call=[&](unsigned schedule,const std::int32_t* first=nullptr,
                     const std::int32_t* second=nullptr,std::size_t count=0) {
    kernels.clear();
    return launch_gradient_family(batch,first,second,count,{},nullptr,
                                                   schedule,1,&output,0);
  };
  assert(call(3)==cudaSuccess);
  assert((kernels==std::vector<PairDomain>{PairDomain::implicit_pairs}));
  assert(call(3,indices,indices,1)==cudaSuccess);
  assert((kernels==std::vector<PairDomain>{PairDomain::explicit_pairs}));
  assert(call(3,indices,nullptr,1)==cudaErrorInvalidValue && kernels.empty());
  assert(call(3,nullptr,nullptr,1)==cudaErrorInvalidValue && kernels.empty());
  assert(call(0)==cudaErrorInvalidValue && kernels.empty());
  assert(call(1)==cudaSuccess && kernels.size()==1);
  assert(call(2)==cudaSuccess && kernels.size()==1);
  batch.ao_shells=nullptr;
  assert(call(3)==cudaErrorInvalidValue && kernels.empty());
  batch.ao_shells=indices;
  batch.atom_offsets=nullptr;
  assert(call(3)==cudaErrorInvalidValue && kernels.empty());
  assert(call(3,indices,indices,1)==cudaSuccess);
  batch.atom_offsets=offsets;
  batch.shell_pair_count=0;
  batch.shell_pair_first=batch.shell_pair_second=nullptr;
  batch.shell_ao_offsets=nullptr;
  assert(call(3)==cudaSuccess && kernels.size()==1);
  assert(call(1)==cudaErrorInvalidValue && kernels.empty());
  batch.nbf=std::numeric_limits<std::int32_t>::max();
  assert(call(3)==cudaErrorInvalidValue && kernels.empty());
  batch.nbf=batch.batch_size=1;
  fail_launch=1;
  assert(call(3)==7 && kernels.size()==1 && sticky_error==7);
  fail_launch=0;
  sticky_error=9;
  assert(call(3)==9 && kernels.size()==1 && sticky_error==9);
}
"""
