"""Thread-emulate the real phased producer/consumer and its ordered fallback."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.method.stationary_cuda import _STATIONARY_SCIENTIFIC_KERNELS
from generativeqc_compiler.xc.grid_native import emit_grid_adjoint, emit_grid_partials
from test_stationary_geometry_kernel_host import PREFIX
from test_stationary_task_work_budget import _block

if TYPE_CHECKING:
    from conftest import NativeCxx


def test_precomputed_status_joins_the_block_vote_before_scratch_reads() -> None:
    """A mutable sticky status must not independently remove CTA participants."""
    kernel = _block(
        _STATIONARY_SCIENTIFIC_KERNELS,
        "__global__ void geometry_cooperative_kernel(",
    )
    entry, point_loop = kernel.split("{", 1)[1].split("for (size_t p = lane;", 1)
    assert not re.search(r"\*\s*error\b", entry)
    vote = _block(point_loop, "if (threadIdx.x == 0) {")
    status = vote.index("!*error")
    assert "control.valid" in vote[:status].splitlines()[-1]
    assert status < vote.index("if (control.valid)")
    assert vote.index("if (control.valid)") < vote.index(
        "reinterpret_cast<const StationaryPointValue*>(ws)"
    )
    publication = point_loop[point_loop.index(vote) + len(vote) :]
    assert publication.index("__syncthreads()") < publication.index(
        "if (!control.valid)"
    )


@pytest.mark.parametrize(("atoms", "aos"), [(12, 0), (12, 96), (12, 1024), (96, 768)])
def test_precomputed_point_preserves_phased_channels_and_fails_closed(
    tmp_path: Path, native_cxx: NativeCxx, atoms: int, aos: int
) -> None:
    """Compare exact channels, seed work, arbitrary AO maps and failure gates."""
    prefix = PREFIX[: PREFIX.index("int local_norm=")].replace(
        "struct { size_t x{}; } threadIdx, blockIdx, blockDim;",
        r"""
#include <atomic>
#include <barrier>
#include <cstring>
#include <limits>
#include <thread>
struct Dimension { size_t x{}; };
thread_local Dimension threadIdx, blockIdx;
Dimension blockDim;
thread_local std::barrier<>* active_barrier;
#define __syncthreads() active_barrier->arrive_and_wait()
#define __shared__ static
double* geometry_pair_storage;
std::atomic<size_t> point_evaluations{0};
""",
    )
    prefix = prefix.replace(
        "const int old=*p; *p=v; return old;", "return std::atomic_ref(*p).exchange(v);"
    ).replace("  return {};", "  ++point_evaluations; return {};")
    begin = _STATIONARY_SCIENTIFIC_KERNELS.index(
        "__device__ bool geometry_point_setup("
    )
    end = _STATIONARY_SCIENTIFIC_KERNELS.index(
        "}  // namespace generativeqc_stationary_cuda", begin
    )
    kernels = _STATIONARY_SCIENTIFIC_KERNELS[begin:end].replace(
        "  extern __shared__ double geometry_pair_storage[];", ""
    )
    source = tmp_path / "precomputed.cpp"
    header = (
        Path(__file__).resolve().parents[2] / "src/dft/stationary_gradient_cuda.cuh"
    ).read_text()
    declaration = re.search(
        r"template\s*<bool precomputed_point>\s*__global__ void geometry_cooperative_kernel\([^;]+;",
        header,
    )
    assert declaration is not None
    source.write_text(
        prefix
        + emit_grid_adjoint()
        + emit_grid_partials(3)
        + "constexpr size_t stationary_becke_retained_max_atoms=32, stationary_becke_pair_tile_rows=4;\n"
        + declaration[0]
        + "\nauto* geometry_entry = &geometry_cooperative_kernel<true>;\n"
        + kernels
        + DRIVER.replace("ATOMS", str(atoms)).replace("AOS", str(aos))
    )
    executable = native_cxx.build_executable(
        [source],
        tmp_path / "precomputed",
        compile_args=["-std=c++20", "-pthread", "-O2", "-ffp-contract=off"],
    )
    subprocess.run([str(executable)], check=True, timeout=30)


DRIVER = r"""
int main() {
  constexpr size_t atoms=ATOMS, aos=AOS, points=5, stride=13;
  constexpr size_t pair_capacity=atoms<=32?atoms*(atoms-1)/2:4*(2*atoms-5)/2;
  static_assert(sizeof(StationaryPointValue)<=3*atoms*sizeof(double));
  std::vector<double> storage(8*pair_capacity+2,987654);
  geometry_pair_storage=storage.data()+1;
  std::vector<double> features(10*points,1), ao(10*points*aos), work(8*points*aos);
  for(size_t coordinate=0;coordinate<ao.size();++coordinate)
    ao[coordinate]=0.4*std::sin(0.17*coordinate);
  for(size_t coordinate=0;coordinate<work.size();++coordinate)
    work[coordinate]=0.3*std::cos(0.13*coordinate);
  std::vector<int64_t> ao_atoms(aos), owners(points);
  std::vector<size_t> active_ids(aos);
  for(size_t index=0;index<aos;++index) {
    ao_atoms[index]=(index*7)%atoms;
    active_ids[index]=(11*(index/2))%aos;
  }
  for(size_t point=0;point<points;++point) owners[point]=(point*7)%atoms;
  for(bool implicit:{false,true}) for(bool external:{false,true}) {
    std::vector<double> weights(points,0.3),raw(points,0.2),seeds(6*stride,0.15);
    std::vector<double> partial(points*9*atoms+2,987654),scratch(points*9*atoms+2,987654);
    std::vector<double> phase_seeds(points+2,987654);
    int error=0,producer_error=0;
    generativeqc::dft::GridTaskView view{points,aos,aos,features.data(),ao.data(),nullptr,
                                       active_ids.data(),&producer_error};
    auto produce=[&] {
      blockDim.x=32;
      for(size_t rank=0;rank<32;++rank) {
        blockIdx.x=0;threadIdx.x=rank;
        geometry_point_kernel(view,implicit?nullptr:owners.data(),4,3,atoms,
            weights.data(),raw.data(),external?seeds.data():nullptr,stride,2,
            scratch.data()+1,phase_seeds.data()+1,&error);
      }
    };
    auto consume=[&]<bool prepared>() {
      blockDim.x=32;
      for(size_t point=0;point<points;++point) {
        std::barrier barrier(std::ptrdiff_t(32));
        std::vector<std::thread> workers;
        for(size_t rank=0;rank<32;++rank) workers.emplace_back([&,rank] {
          threadIdx.x=rank;blockIdx.x=point;active_barrier=&barrier;
          geometry_cooperative_kernel<prepared>(view,work.data(),ao_atoms.data(),
              implicit?nullptr:owners.data(),4,3,nullptr,atoms,weights.data(),raw.data(),
              external?seeds.data():nullptr,stride,2,points,partial.data()+1,
              scratch.data()+1,nullptr,&error,phase_seeds.data()+1);
        });
        for(auto& worker:workers) worker.join();
      }
    };
    point_evaluations=0;
    consume.template operator()<false>();
    if(error || point_evaluations!=(external?0:points)) return 1;
    const auto expected=partial,expected_seeds=phase_seeds;
    std::fill(partial.begin(),partial.end(),987654);
    point_evaluations=0;
    produce();consume.template operator()<true>();
    if(error || point_evaluations!=(external?0:points)) return 2;
    if(std::memcmp(partial.data(),expected.data(),partial.size()*sizeof(double)) ||
       phase_seeds!=expected_seeds) return 3;
    if(scratch.front()!=987654 || scratch.back()!=987654 ||
       storage.front()!=987654 || storage.back()!=987654) return 4;
    for(size_t invalid=0;invalid<7;++invalid) {
      const auto old_raw=raw,old_weights=weights,old_seeds=seeds;
      const auto old_owners=owners;
      error=0;
      if(invalid==0) producer_error=1;
      if(invalid==1) raw.back()=std::numeric_limits<double>::infinity();
      if(invalid==2) weights.back()=std::numeric_limits<double>::quiet_NaN();
      if(invalid==3) { if(implicit) continue; owners.back()=-1; }
      if(invalid==4) { if(!external) continue; seeds[5*stride+points+1]=std::numeric_limits<double>::infinity(); }
      if(invalid==5) error=1;
      if(invalid==6) { if(!aos) continue; view.nao=0; }
      produce();consume.template operator()<true>();
      if(!error) return 5;
      std::vector<double> result(9*atoms,123);
      for(size_t coordinate=0;coordinate<9*atoms;++coordinate) {
        blockIdx.x=coordinate/32;threadIdx.x=coordinate%32;
        geometry_reduce(partial.data()+1,atoms,points,result.data(),&error);
      }
      for(double value:result) if(value!=123) return 6;
      raw=old_raw;weights=old_weights;seeds=old_seeds;owners=old_owners;
      producer_error=0;view.nao=aos;
    }
  }
  return 0;
}
"""
