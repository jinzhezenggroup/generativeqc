"""Execute emitted normalization for rectangular and nonuniform atom grids."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from generativeqc_compiler.xc.quadrature_cuda import emit_quadrature_cuda
from test_stationary_task_work_budget import _block

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import NativeCxx


def test_normalize_preserves_owner_weights_and_finite_boundaries(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    kernel = _block(emit_quadrature_cuda(), "__global__ void normalize_kernel(")
    source = tmp_path / "normalize.cpp"
    source.write_text(
        r"""
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#define __global__
using std::isfinite;
struct Dimension { size_t x; } blockIdx{0},threadIdx{0},blockDim{3},gridDim{2};
int atomicExch(int* pointer,int value) { const int old=*pointer; *pointer=value; return old; }
"""
        + kernel
        + r"""
void launch(size_t begin,size_t count,size_t per_atom,const double* logs,
            double* weights,int* invalid,const uint64_t* offsets) {
  for(blockIdx.x=0;blockIdx.x<gridDim.x;++blockIdx.x)
    for(threadIdx.x=0;threadIdx.x<blockDim.x;++threadIdx.x)
      normalize_kernel(begin,count,per_atom,4,logs,weights,invalid,offsets);
}
int main() {
  // Nonuniform counts include an empty atom; no rectangular product can
  // recover this ownership map. Exercise every bounded tile across boundaries.
  const uint64_t offsets[]{0,2,2,5,8};
  const size_t nonuniform_owners[]{0,0,2,2,2,3,3,3};
  for(bool nonuniform:{false,true}) for(size_t begin=0;begin<8;++begin)
    for(size_t count=1;count<=8-begin;++count) {
      std::vector<double> logs(4*count),weights(count+2,987654),expected(count);
      for(size_t p=0;p<count;++p) {
        // Signed and zero raw weights remain valid for Lebedev rules.
        weights[p+1]=p%3==0?-0.125:p%3==1?0:0.75;
        const size_t owner=nonuniform?nonuniform_owners[begin+p]:(begin+p)/2;
        expected[p]=weights[p+1]*double(owner+1)/10;
        for(size_t atom=0;atom<4;++atom)
          logs[atom*count+p]=std::log(double(atom+1))+(p%2?750:-750);
      }
      int invalid=0;
      launch(begin,count,nonuniform?0:2,logs.data(),weights.data()+1,&invalid,
             nonuniform?offsets:nullptr);
      assert(invalid==0 && weights.front()==987654 && weights.back()==987654);
      for(size_t p=0;p<count;++p)
        assert(isfinite(weights[p+1]) && std::abs(weights[p+1]-expected[p])<3e-13);
    }
  for(int mode=0;mode<4;++mode) {
    double logs[4]{},weight=1;
    int invalid=0;
    if(mode==0) for(double& value:logs) value=-INFINITY;
    if(mode==1) logs[0]=std::numeric_limits<double>::quiet_NaN();
    if(mode==2) weight=INFINITY;
    if(mode==3) invalid=1;
    launch(0,1,2,logs,&weight,&invalid,nullptr);
    assert(invalid==1);
  }
}
"""
    )
    binary = native_cxx.build_executable(
        [source], tmp_path / "normalize", compile_args=["-std=c++17", "-O2"]
    )
    subprocess.run([str(binary)], check=True, timeout=10)
