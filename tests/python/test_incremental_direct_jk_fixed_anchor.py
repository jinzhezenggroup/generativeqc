"""Execute HF device-controller bodies on the host; not a CUDA performance gate."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def _kernel(source: str, name: str) -> str:
    """Extract a production kernel without replacing its controller logic."""
    begin = source.index(f"__global__ void {name}(")
    end = source.index("{", begin) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end].replace("__global__ ", "", 1)


@pytest.fixture(scope="module")
def anchor_probe(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> Path:
    """Compile actual kernels with sequential thread dispatch and host atomics."""
    source = (ROOT / "src/scf/cuda/scf_matrix_kernels.cu").read_text()
    unit = r"""
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
using std::isfinite;
struct { unsigned x; } blockIdx{0}, blockDim{128}, threadIdx{0};
long long __double_as_longlong(double value) { return std::bit_cast<long long>(value); }
void atomicMax(unsigned long long* target, unsigned long long value) {
  unsigned long long previous=0;
  std::memcpy(&previous,target,sizeof(previous));
  const auto maximum=std::max(previous,value);
  std::memcpy(target,&maximum,sizeof(maximum));
}
void require(bool condition, const char* message) {
  if(!condition) throw std::runtime_error(message);
}
"""
    unit += _kernel(source, "prepare_incremental_direct_jk_kernel")
    unit += _kernel(source, "finalize_incremental_direct_jk_kernel")
    unit += r"""
int main(int argc, char** argv) {
  if(argc!=4) return 99;
  try {
    const bool fixed=std::string(argv[1])=="fixed";
    const unsigned interval=std::stoul(argv[2]);
    const int spins=std::stoi(argv[3]);
    constexpr int batch=2, nbf=2, matrix=nbf*nbf;
    const unsigned elements=spins*matrix, total=batch*elements;
    std::array<double,16> density{},anchor_density{},anchor_fock{},delta{},fock{};
    std::array<double,8> hcore{};
    std::array<double,2> maximum{},rms{};
    std::array<std::uint32_t,2> updates{0xffffffffU,0xffffffffU};
    std::array<std::uint8_t,2> active{1,0},full{};
    std::array<unsigned long long,2> full_count{},delta_count{};
    for(unsigned element=0;element<total;++element) {
      density[element]=(element%2 ? -1.0 : 1.0)*(1.0+element/8.0);
      anchor_density[element]=-42;
      anchor_fock[element]=-43;
      delta[element]=-44;
      fock[element]=-45;
    }
    for(unsigned element=0;element<hcore.size();++element) hcore[element]=0.5+element/16.0;
    constexpr double cutoff=0.125;
    const auto screened=[](double value) { return std::abs(value)<cutoff ? 0.0 : value; };
    unsigned reference_full=0,reference_delta=0,since_full=0;
    std::array<double,8> full_density{},full_operator{};
    for(unsigned iteration=0;iteration<20;++iteration) {
      // One nonfinite preceding RMS must force a full refresh, even with no cadence.
      rms[0]=iteration==11 ? std::numeric_limits<double>::quiet_NaN() : 0.0;
      const bool rebuild=iteration==0 || iteration==11 ||
                         (interval!=0 && since_full>=interval);
      if(rebuild) {++reference_full;since_full=0;} else {++reference_delta;++since_full;}
      for(threadIdx.x=0;threadIdx.x<total;++threadIdx.x)
        prepare_incremental_direct_jk_kernel(batch,spins,nbf,interval,1.0,rms.data(),
            density.data(),hcore.data(),active.data(),anchor_density.data(),anchor_fock.data(),
            delta.data(),updates.data(),full.data(),full_count.data(),delta_count.data(),
            maximum.data());
      require(full[0]==rebuild,"full refresh disagreed with cadence/RMS policy");
      for(unsigned element=0;element<elements;++element) {
        if(rebuild) {
          full_density[element]=density[element];
          full_operator[element]=screened(2*density[element]);
        }
        fock[element]=hcore[element%matrix]+screened(2*delta[element]);
      }
      for(threadIdx.x=0;threadIdx.x<total;++threadIdx.x)
        finalize_incremental_direct_jk_kernel(batch,spins,nbf,fixed,density.data(),hcore.data(),
            active.data(),anchor_density.data(),anchor_fock.data(),fock.data(),updates.data(),
            full.data());
      for(unsigned element=0;element<elements;++element) {
        if(fixed) {
          const double expected=hcore[element%matrix]+full_operator[element]+
              screened(2*(density[element]-full_density[element]));
          require(fock[element]==expected,"screened errors accumulated beyond full+current delta");
          require(anchor_density[element]==full_density[element],"screened delta replaced full anchor");
          require(std::abs(fock[element]-hcore[element%matrix]-2*density[element])<2*cutoff,
                  "fixed-anchor screened error grew with delta count");
        } else {
          require(anchor_density[element]==density[element] && anchor_fock[element]==fock[element],
                  "rolling exact-linear anchor stopped advancing");
        }
        density[element]+=(element%2 ? -1.0 : 1.0)/32;
      }
      require(updates[0]==since_full,"delta update count changed with anchor mode");
      require(anchor_density[elements]==-42 && anchor_fock[elements]==-43 &&
                  delta[elements]==-44 && fock[elements]==-45 && updates[1]==0xffffffffU,
              "inactive spin/batch item mutated");
    }
    require(full_count[0]==reference_full && delta_count[0]==reference_delta &&
                full_count[1]==0 && delta_count[1]==0,
            "full/delta census lost actual provider applications");
    if(!fixed && interval==0)
      require(std::abs(fock[0]-hcore[0]-2*(density[0]-1.0/32))>=cutoff,
              "screened rolling-anchor negative control did not accumulate omissions");
  } catch(const std::exception& error) {std::cerr<<error.what();return 1;}
}
"""
    directory = tmp_path_factory.mktemp("incremental-fixed-anchor")
    cpp, executable = directory / "probe.cpp", directory / "probe"
    cpp.write_text(unit)
    native_cxx.build_executable(
        [cpp],
        executable,
        compile_args=["-std=c++20", "-O2", "-fsanitize=undefined"],
        link_args=["-fsanitize=undefined"],
    )
    return executable


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("interval", [0, 1, 3, 8])
@pytest.mark.parametrize("mode", ["fixed", "rolling"])
def test_device_anchor_reconstruction_and_screening_error(
    anchor_probe: Path, mode: str, interval: int, spins: int
) -> None:
    """Gate cadence, error depth, spin layout, inactive items and RMS refreshes."""
    result = subprocess.run(
        [str(anchor_probe), mode, str(interval), str(spins)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
