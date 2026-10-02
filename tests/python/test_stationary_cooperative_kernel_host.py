"""Thread-emulate the emitted cooperative kernel with real generated Becke math."""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest
from generativeqc_compiler.method.stationary_cuda import _STATIONARY_SCIENTIFIC_KERNELS
from generativeqc_compiler.xc.grid_native import emit_grid_adjoint, emit_grid_partials
from test_stationary_geometry_kernel_host import PREFIX


def test_emitted_cooperative_kernel_routes_tails_and_sticky_failure(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    prefix = PREFIX[: PREFIX.index("int local_norm=")]
    prefix = prefix.replace(
        "struct { size_t x{}; } threadIdx, blockIdx, blockDim;",
        r"""
#include <atomic>
#include <barrier>
#include <thread>
struct Dimension { size_t x{}; };
thread_local Dimension threadIdx, blockIdx;
Dimension blockDim;
thread_local std::barrier<>* active_barrier;
#define __syncthreads() active_barrier->arrive_and_wait()
#define __shared__ static
double* geometry_pair_storage;
""",
    )
    prefix = prefix.replace(
        "const int old=*p; *p=v; return old;", "return std::atomic_ref(*p).exchange(v);"
    )
    prefix = prefix.replace(
        "*error=1; return fallback;",
        "std::atomic_ref(*error).store(1); return fallback;",
    )
    begin = _STATIONARY_SCIENTIFIC_KERNELS.index("__device__ bool geometry_point_ao(")
    end = _STATIONARY_SCIENTIFIC_KERNELS.index(
        "}  // namespace generativeqc_stationary_cuda", begin
    )
    kernels = _STATIONARY_SCIENTIFIC_KERNELS[begin:end].replace(
        "  extern __shared__ double geometry_pair_storage[];", ""
    )
    source = tmp_path / "kernel.cpp"
    source.write_text(
        prefix
        + emit_grid_adjoint()
        + emit_grid_partials(3)
        + kernels
        + r"""
int main() {
  constexpr size_t na=12,n=2,np=17,pairs=na*(na-1)/2;
  double centers[3*na];
  for(size_t a=0;a<na;++a) {
    centers[3*a]=0.7*a; centers[3*a+1]=0.4*std::sin(a); centers[3*a+2]=0.3*std::cos(a);
  }
  std::vector<generativeqc_grid_adjoint::CenterPair> geometry(pairs);
  if(!generativeqc_grid_adjoint::prepare_center_geometry(centers,na,1e-12,geometry.data(),
                                                         local_norm,local_ratio_geometry)) return 1;
  const int64_t ao_atoms[n]{0,na-1};
  for(bool cached:{false,true}) for(bool implicit:{false,true}) for(bool external:{false,true})
  for(size_t capacity:{size_t(1),size_t(7),size_t(17)}) for(size_t points:{size_t(0),size_t(1),np}) {
    const size_t lanes=std::min(capacity,points);
    std::vector<double> partial(capacity*9*na+2,987654),scratch(capacity*9*na+2,987654);
    std::vector<double> storage(8*pairs+2,987654);
    geometry_pair_storage=storage.data()+1;
    std::vector<double> xyz(3*points),features(10*points,1),ao(10*points*n,0.5),work(8*points*n,0.75);
    std::vector<double> weights(points,0.3),raw(points,0.2),seeds(6*(np+7),0.15);
    std::vector<int64_t> owners(points);
    for(size_t p=0;p<points;++p) {
      owners[p]=implicit?(p+4)/3:(7*p)%na;
      xyz[3*p]=0.1+0.3*p; xyz[3*p+1]=0.7; xyz[3*p+2]=-0.8;
    }
    const auto* center_pairs=cached?geometry.data():nullptr;
    int error=0,producer_error=0;
    generativeqc::dft::GridTaskView view{points,n,n,features.data(),ao.data(),xyz.data(),nullptr,&producer_error};
    auto invoke=[&](bool cooperative,size_t lane,size_t rank) {
      blockIdx.x=cooperative?lane:lane/32; threadIdx.x=cooperative?rank:lane%32;
      auto kernel=cooperative?geometry_cooperative_kernel:geometry_kernel;
      kernel(view,work.data(),ao_atoms,implicit?nullptr:owners.data(),4,3,centers,na,
             weights.data(),raw.data(),external?seeds.data():nullptr,np+7,2,
             lanes,partial.data()+1,scratch.data()+1,center_pairs,&error);
    };
    auto reduce=[&] {
      std::vector<double> output(9*na);
      for(size_t k=0;k<9*na;++k) {
        blockIdx.x=k/32; threadIdx.x=k%32;
        geometry_reduce(partial.data()+1,na,lanes,output.data(),&error);
      }
      return output;
    };
    blockDim.x=32;
    for(size_t lane=0;lane<lanes;++lane) invoke(false,lane,0);
    const auto expected=reduce();
    std::fill(partial.begin(),partial.end(),987654);
    auto execute=[&] {
      for(size_t lane=0;lane<lanes;++lane) {
        std::barrier barrier(32);
        std::vector<std::thread> workers;
        for(size_t rank=0;rank<32;++rank) workers.emplace_back([&,rank] {
          active_barrier=&barrier; invoke(true,lane,rank);
        });
        for(auto& worker:workers) worker.join();
      }
    };
    execute();
    const auto actual=reduce();
    if(error) return 2;
    for(size_t k=0;k<9*na;++k) if(std::abs(expected[k]-actual[k])>2e-11) return 3;
    if(partial.front()!=987654 || partial.back()!=987654 || scratch.front()!=987654 ||
       scratch.back()!=987654 || storage.front()!=987654 || storage.back()!=987654) return 4;
    if(!points) continue;
    // Invalid input late in a worker never publishes any partial output.
    for(int invalid=0;invalid<5;++invalid) {
      const auto old_xyz=xyz,old_raw=raw,old_seeds=seeds;
      if(invalid==0) std::copy(centers,centers+3,xyz.end()-3);
      if(invalid==1) raw.back()=std::numeric_limits<double>::quiet_NaN();
      if(invalid==2) producer_error=1;
      if(invalid==3) view.nao=1;
      if(invalid==4) { if(!external) continue; seeds[5*(np+7)+points+1]=std::numeric_limits<double>::infinity(); }
      error=0; execute();
      const auto failed=reduce();
      if(!error) return 5;
      for(double value:failed) if(value!=0) return 6;
      xyz=old_xyz; raw=old_raw; seeds=old_seeds; producer_error=0; view.nao=n;
      // Rebind potentially replaced vector storage before the next replay.
      view.points=xyz.data();
    }
  }
  return 0;
}
"""
    )
    binary = tmp_path / "kernel"
    process = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-pthread",
            "-O2",
            "-ffp-contract=off",
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    result = subprocess.run(
        [str(binary)], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
