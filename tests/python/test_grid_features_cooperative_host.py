"""Thread-emulate the emitted warp schedule against independent long-double sums."""

from __future__ import annotations

import re
import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest
from generativeqc_compiler.dft.ao_cuda import emit_grid_scientific_kernels
from generativeqc_compiler.dft.feature_policy import emit_feature_policy


@pytest.fixture(scope="module")
def feature_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = emit_grid_scientific_kernels()
    begin = source.index("__global__ void feature_kernel(")
    schedule_begin = source.index("inline void scheduled_grid_features(", begin)
    end = source.index("// Reduce one occupied tile", schedule_begin)
    kernels = source[begin:schedule_begin]
    schedule = re.sub(
        r"(\w+)<<<(.*?)>>>\(", r"launch(\1, \2, ", source[schedule_begin:end]
    )
    policy = emit_feature_policy().replace(
        "unsigned mask) {", "unsigned mask) { ++bilinear_calls;"
    )
    prefix = r"""
#include <algorithm>
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <thread>
#include <vector>
#define __global__
using I = std::int64_t;
using cudaStream_t = int;
struct Dimension { I x{}; };
thread_local Dimension blockIdx, threadIdx;
Dimension blockDim, gridDim;
std::barrier warp_barrier(32);
double shuffle_values[32];
std::atomic<I> bilinear_calls{};
I launched_work{};
double __shfl_down_sync(unsigned mask, double value, int offset) {
  if (mask != 0xffffffffu) std::abort();
  const I lane = threadIdx.x % 32;
  shuffle_values[lane] = value;
  warp_barrier.arrive_and_wait();
  const double result = shuffle_values[lane + offset < 32 ? lane + offset : lane];
  warp_barrier.arrive_and_wait();
  return result;
}
double finite(double value, int* error, int node) {
  if (!std::isfinite(value) && !*error) *error = node + 1;
  return value;
}
I blocks(I work, I threads) {
  launched_work = work;
  return std::min(I(1), (work + threads - 1) / threads);
}
"""
    launch = r"""
template<class Kernel, class... Args>
void launch(Kernel kernel, I grid, I threads, int shared, int stream, Args... args) {
  if (threads != 128 || shared || stream != 7) std::abort();
  gridDim.x = grid;
  blockDim.x = threads;
  for (I block = 0; block < grid; ++block) {
    if (kernel == cooperative_feature_kernel) {
      // Warps have no cross-warp state: emulate one at a time, retaining
      // the actual block/thread indices and grid-stride loops.
      for (I warp = 0; warp < threads / 32; ++warp) {
        std::vector<std::jthread> workers;
        for (I lane = 0; lane < 32; ++lane)
          workers.emplace_back([=] {
            blockIdx.x = block;
            threadIdx.x = warp * 32 + lane;
            kernel(args...);
          });
      }
    } else {
      blockIdx.x = block;
      for (threadIdx.x = 0; threadIdx.x < threads; ++threadIdx.x) kernel(args...);
    }
  }
}
"""
    main = r"""
int probe(I nao, I npoint, unsigned mask, int mode) {
  const I stride = nao * npoint;
  std::vector<double> ao(4*stride+2, 12345.25), work(8*stride+2, 12345.25);
  const double poison = std::numeric_limits<double>::quiet_NaN();
  for (I index = 0; index < 4*stride; ++index)
    ao[index+1] = mode == 1 ? 0.0 : 0.4*std::sin(0.17*index);
  for (I index = 0; index < 8*stride; ++index)
    work[index+1] = mode == 1 ? 0.0 : 0.3*std::cos(0.13*index);
  // Unrequested jet slots deliberately contain NaNs. Masked work must never
  // enter the arithmetic, including value jets in a tau-only request.
  if (!(mask & 14)) std::fill(ao.begin()+1+stride,ao.end()-1,poison);
  for (I spin = 0; spin < 2; ++spin) {
    if (!(mask & 7))
      std::fill(work.begin()+1+spin*4*stride,work.begin()+1+(spin*4+1)*stride,poison);
    if (!(mask & 8))
      std::fill(work.begin()+1+(spin*4+1)*stride,work.begin()+1+(spin*4+4)*stride,poison);
  }
  if (mode == 2 && stride) work[1] = poison;
  std::vector<double> output(13*npoint+2,12345.25);
  std::fill(output.begin()+1,output.end()-1,0.0);
  int error = mode == 3 ? 9 : 0;
  bilinear_calls = 0;
  scheduled_grid_features(7,ao.data()+1,work.data()+1,npoint,nao,output.data()+1,&error,mask);
  if (launched_work != npoint*(nao >= 32 ? 32 : 1)) return 1;
  if (bilinear_calls != 2*nao*npoint) return 2;
  if (output.front() != 12345.25 || output.back() != 12345.25) return 3;
  if (mode == 2 && stride) return error == 2 ? 0 : 4;
  if (error != (mode == 3 ? 9 : 0)) return 5;
  for (I point = 0; point < npoint; ++point) {
    long double expected[13]{};
    for (I spin = 0; spin < 2; ++spin)
      for (I ao_index = 0; ao_index < nao; ++ao_index) {
        const I index = 1+point*nao+ao_index;
        const I panel = index+spin*4*stride;
        if (mask & 1) expected[spin*5] += (long double)ao[index]*work[panel];
        for (I axis = 0; axis < 3; ++axis) {
          if (mask & 6)
            expected[spin*5+axis+1] += 2.0L*ao[index+(axis+1)*stride]*work[panel];
          if (mask & 8)
            expected[spin*5+4] += 0.5L*ao[index+(axis+1)*stride]*work[panel+(axis+1)*stride];
        }
      }
    if (mask & 4)
      for (I axis = 0; axis < 3; ++axis) {
        expected[10] += expected[axis+1]*expected[axis+1];
        expected[11] += expected[axis+1]*expected[axis+6];
        expected[12] += expected[axis+6]*expected[axis+6];
      }
    for (I feature = 0; feature < 13; ++feature) {
      const auto difference = std::abs((long double)output[1+feature*npoint+point]-expected[feature]);
      if (difference > 1e-11L*(1+std::abs(expected[feature]))) return 6;
    }
  }
  return 0;
}
int main(int argc, char** argv) {
  if (argc != 2) return 10;
  const I nao = std::atoi(argv[1]);
  for (unsigned mask = 1; mask < 16; ++mask)
    if (const int status = probe(nao,3,mask,0)) return 10+status;
  for (int mode = 0; mode < 4; ++mode)
    if (const int status = probe(nao,9,15,mode)) return 20+status;
  return probe(nao,0,15,0);
}
"""
    directory = tmp_path_factory.mktemp("grid-feature-cooperative")
    path, executable = directory / "probe.cpp", directory / "probe"
    path.write_text(
        prefix
        + "namespace generativeqc_grid_policy {\n"
        + policy
        + "}\n"
        + kernels
        + launch
        + schedule
        + main
    )
    result = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            "-pthread",
            str(path),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.parametrize("nao", [0, 1, 31, 32, 33, 65, 768])
def test_cooperative_grid_features_preserve_masks_work_and_oracle(
    feature_probe: Path, nao: int
) -> None:
    result = subprocess.run(
        [str(feature_probe), str(nao)],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, (result.returncode, result.stderr)
