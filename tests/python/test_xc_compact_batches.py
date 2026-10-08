"""Host admission and execution of generated compact batching (no GPU probe).

The CPU CUDA-block simulator executes the actual shared-memory density/Vxc and
ordered-scatter bodies. Independent scalar bilinears check maps, compact tails,
spin/jet strides and overlapping output ownership. This is not GPU qualification.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from generativeqc_compiler.dft.ao_cuda import _NATIVE_XC_CONTRACTION_KERNELS
from generativeqc_compiler.dft.feature_policy import emit_feature_policy
from generativeqc_compiler.dft.xc_contraction_cuda import (
    DEFAULT_XC_MATRIX_SCHEDULE,
    _emit_panels,
    _emit_tiled,
)
from generativeqc_compiler.dft.xc_point_batch_cuda import (
    emit_native_xc_point_batch_plan,
)
from generativeqc_compiler.dft.xc_tile_batch_cuda import emit_native_xc_tile_batches

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def definition(source: str, signature: str) -> str:
    """Extract one emitter definition so the tested arithmetic is not copied."""
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def compact_types() -> str:
    """Use the actual public offsets/arena metadata in standalone native probes."""
    source = (ROOT / "src/dft/cuda_xc.hpp").read_text()
    return "\n".join(
        definition(source, f"struct {name} {{") + ";"
        for name in ("CudaXcCompactTile", "CudaXcPointBatchPlan")
    )


def test_compact_admission_charges_every_slot_and_preserves_fallback(
    tmp_path: Path,
    native_cxx: NativeCxx,
) -> None:
    """Peak panels, local matrices and all descriptors share the same byte cap."""
    source = tmp_path / "compact-plan.cpp"
    source.write_text(
        _PLAN_HEADER + compact_types() + emit_native_xc_point_batch_plan() + _PLAN_MAIN
    )
    binary = tmp_path / "compact-plan"
    native_cxx.build_executable([source], binary, compile_args=("-std=c++17", "-O2"))
    subprocess.run([str(binary)], check=True, timeout=30)


def test_emitted_compact_arithmetic_matches_independent_bilinears(
    tmp_path: Path,
    native_cxx: NativeCxx,
) -> None:
    """Run the real generated bodies with CPU barriers, not a rewritten schedule."""
    features = definition(
        _NATIVE_XC_CONTRACTION_KERNELS,
        "template<bool Cooperative>\n__device__ void density_features_body",
    )
    matrices = _emit_tiled(DEFAULT_XC_MATRIX_SCHEDULE).split(
        "// The compiler owns schedule admission"
    )[0]
    # Count executed reduction steps without replacing their scientific math.
    density_step = "value += d[x][k]*a[y][k];"
    potential_step = "value += am[k][x]*wn[k][y]+wm[k][x]*an[k][y];"
    assert density_step in matrices and potential_step in matrices
    matrices = matrices.replace(
        density_step,
        "{density_terms.fetch_add(1,std::memory_order_relaxed);" + density_step + "}",
    ).replace(
        potential_step,
        "{potential_terms.fetch_add(1,std::memory_order_relaxed);"
        + potential_step
        + "}",
    )
    batch = emit_native_xc_tile_batches().split("inline void launch_batch_density")[0]
    source = tmp_path / "compact-numerics.cpp"
    source.write_text(
        _SIMULATOR
        + compact_types()
        + "namespace generativeqc_grid_policy {\n"
        + emit_feature_policy(device=True)
        + "}\n"
        + "namespace generativeqc::dft::cuda_xc_detail { namespace {\n"
        + features
        + _emit_panels()
        + matrices
        + "}}\n"
        + batch
        + "}}\n"
        + _NUMERICS
    )
    binary = tmp_path / "compact-numerics"
    native_cxx.build_executable(
        [source],
        binary,
        compile_args=("-std=c++20", "-O2", "-ffp-contract=off", "-pthread"),
        link_args=("-pthread",),
    )
    subprocess.run([str(binary)], check=True, timeout=120)


_PLAN_HEADER = r"""
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
#include <cassert>
enum class CudaXcAoPrecision { Fp64, Fp32ComputeFp64Storage };
struct CudaXcLayout {
  std::size_t npoint{235}, tile_points{64}, nao{80}, jets{4}, spins{2}, feature_terms{5}, work_jets{4};
  bool local_ao{true}, response{};
  CudaXcAoPrecision ao_precision{CudaXcAoPrecision::Fp64};
};
"""

_PLAN_MAIN = r"""
int main() {
  CudaXcLayout layout;
  const std::vector<std::size_t> offsets{0,0,32,80,120};
  const auto plain=prepare_point_batch_plan(layout,offsets,4,1<<20);
  const auto plan=prepare_point_batch_plan(layout,offsets,4,1<<20,true);
  assert(plan.compact && plan.tiles==4 && plan.ao_elements==4*(64*32+64*48+43*40));
  assert(plan.work_elements==8*(64*32+64*48+43*40));
  assert(plan.potential_elements==2*(32*32+48*48+40*40));
  assert(plan.descriptor_bytes==4*sizeof(CudaXcCompactTile));
  assert(plan.compact_groups==1 && plan.compact_tiles==4 && plan.compact_nonempty_tiles==3);
  assert(plan.device_bytes==(plan.ao_elements+2*plan.feature_elements+plan.total_elements+
      plan.work_elements+plan.potential_elements)*sizeof(double)+plan.descriptor_bytes);
  assert(prepare_point_batch_plan(layout,offsets,4,plan.device_bytes,true).tiles==4);
  assert(prepare_point_batch_plan(layout,offsets,4,plan.device_bytes-1,true).tiles<4);
  const auto fallback=prepare_point_batch_plan(layout,offsets,4,plain.device_bytes,true);
  assert(!fallback.compact && fallback.tiles==plain.tiles && fallback.device_bytes==plain.device_bytes);
  for (auto budget : {std::size_t{0},std::size_t{1},std::size_t{127},plan.device_bytes-1})
    assert(prepare_point_batch_plan(layout,offsets,4,budget,true).device_bytes<=budget);
  layout.npoint=193;
  assert(!prepare_point_batch_plan(layout,offsets,4,1<<20,true).compact);
  layout.npoint=235;
  assert(!prepare_point_batch_plan(layout,{0,0,31,79,119},4,1<<20,true).compact);
  layout.response=true;
  assert(prepare_point_batch_plan(layout,offsets,4,1<<20,true).tiles==1);
  layout.response=false;
  layout.ao_precision=CudaXcAoPrecision::Fp32ComputeFp64Storage;
  assert(prepare_point_batch_plan(layout,offsets,4,1<<20,true).tiles==1);
  layout.ao_precision=CudaXcAoPrecision::Fp64;
  // One unsupported group must not disable independent small groups, nor
  // reserve compact work/matrix storage for its large AO domain.
  layout.npoint=512;
  layout.nao=256;
  const std::vector<std::size_t> mixed{0,0,32,80,120,120,320,360,400};
  const auto partial=prepare_point_batch_plan(layout,mixed,4,2<<20,true);
  assert(partial.compact && partial.tiles==4 && partial.compact_groups==1);
  assert(partial.compact_tiles==4 && partial.compact_nonempty_tiles==3);
  assert(compact_point_batch_admitted(layout,mixed,0,4));
  assert(!compact_point_batch_admitted(layout,mixed,4,8));
  assert(partial.work_elements==8*64*(32+48+40));
  assert(partial.potential_elements==2*(32*32+48*48+40*40));
  layout.spins=std::numeric_limits<std::size_t>::max();
  try { prepare_point_batch_plan(layout,mixed,4,1<<20,true); assert(false); }
  catch (const std::invalid_argument&) {}
}
"""

_SIMULATOR = r"""
#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>
using I=std::int64_t;
struct Dim { unsigned x{1},y{1},z{1}; };
thread_local Dim threadIdx,blockIdx;
Dim blockDim,gridDim;
std::barrier<>* block_barrier;
std::array<std::unique_ptr<std::barrier<>>,8> warp_barriers;
std::array<double,256> shuffle_values;
std::atomic<std::size_t> density_terms{},potential_terms{};
#define __global__
#define __device__
#define __shared__ static
void __syncthreads() { block_barrier->arrive_and_wait(); }
double __shfl_down_sync(unsigned,double value,int offset) {
  const unsigned lane=threadIdx.x, warp=lane/32;
  shuffle_values[lane]=value;
  warp_barriers[warp]->arrive_and_wait();
  const double result=lane%32+offset<32 ? shuffle_values[lane+offset] : value;
  warp_barriers[warp]->arrive_and_wait();
  return result;
}
float __double2float_rn(double value) { return float(value); }
float __fadd_rn(float left,float right) { return left+right; }
float __fmul_rn(float left,float right) { return left*right; }
double __dadd_rn(double left,double right) { return left+right; }
double finite(double value,int* error,int node) {
  if (!std::isfinite(value)) std::atomic_ref<int>(*error).store(node+1);
  return value;
}
template<class Body> void launch(Dim grid,Dim block,Body body) {
  gridDim=grid; blockDim=block;
  const unsigned workers=block.x*block.y;
  std::barrier barrier(workers);
  block_barrier=&barrier;
  for (auto& warp : warp_barriers) warp=std::make_unique<std::barrier<>>(32);
  std::vector<std::thread> threads;
  for (unsigned lane=0;lane<workers;++lane)
    threads.emplace_back([&,lane] {
      threadIdx={lane%block.x,lane/block.x,0};
      for (unsigned plane=0;plane<grid.z;++plane)
        for (unsigned row=0;row<grid.y;++row)
          for (unsigned column=0;column<grid.x;++column) {
            blockIdx={column,row,plane};
            barrier.arrive_and_wait();
            body();
            barrier.arrive_and_wait();
          }
    });
  for (auto& thread : threads) thread.join();
}
void close(double value,double expected) {
  assert(std::isfinite(value) && std::abs(value-expected)<=2e-11+2e-12*std::abs(expected));
}
"""

_NUMERICS = r"""
using namespace generativeqc::dft::cuda_xc_detail;
int main() {
  constexpr I full_n=64,spins=2,jets=4,terms=5,work_jets=4;
  std::vector<std::size_t> ids;
  for (std::size_t ao=0;ao<32;++ao) ids.push_back(2*ao);
  for (std::size_t ao=16;ao<63;++ao) ids.push_back(ao);
  for (std::size_t ao=16;ao<63;++ao) ids.push_back(ao);
  std::vector<CudaXcCompactTile> tiles{{0,64,0,0,0,0,0},{64,64,32,0,0,0,0},
      {128,64,47,32,4*64*32,8*64*32,2*32*32},
      {192,33,47,79,4*64*(32+47),8*64*(32+47),2*(32*32+47*47)}};
  const I ao_size=4*(64*32+97*47),work_size=8*(64*32+97*47);
  std::vector<double> ao(ao_size),work(work_size),features(spins*terms*225),coefficients(features.size()),
      weights(225),point_totals(3*225),local(2*(32*32+2*47*47)),potential(spins*full_n*full_n),
      expected(potential.size()),density(potential.size());
  for (std::size_t index=0;index<ao.size();++index) ao[index]=0.03*std::sin(double(index+1));
  for (I spin=0;spin<spins;++spin)
    for (I row=0;row<full_n;++row)
      for (I col=0;col<full_n;++col)
        density[(spin*full_n+row)*full_n+col]=(row==col ? 0.8 : 0.0)+0.0009765625*(row+1)*(col+1)/(spin+1);
  for (std::size_t index=0;index<coefficients.size();++index)
    coefficients[index]=0.07*std::cos(double(index+1));
  for (I point=0;point<225;++point) weights[point]=0.1+0.001*point;
  for (std::size_t index=0;index<point_totals.size();++index) point_totals[index]=0.01*index;
  int error=0;
  launch({3,4,32},{16,16,1},[&] {
    batch_density_products(tiles.data(),density.data(),ao.data(),work.data(),full_n,spins,
                            work_jets,ids.data(),&error);
  });
  // Dense scalar D*AO is an independent reference for every selected row/jet.
  for (const auto& tile : tiles)
    for (I spin=0;spin<spins;++spin)
      for (I jet=0;jet<jets;++jet)
        for (I point=0;point<I(tile.count);++point)
          for (I mu=0;mu<I(tile.active);++mu) {
            double value=0.0;
            for (I nu=0;nu<I(tile.active);++nu)
              value+=density[(spin*full_n+ids[tile.map_offset+mu])*full_n+ids[tile.map_offset+nu]]*
                  ao[tile.ao_offset+(jet*tile.count+point)*tile.active+nu];
            close(work[tile.work_offset+((spin*jets+jet)*tile.count+point)*tile.active+mu],value);
          }
  launch({32,4,1},{128,1,1},[&] {
    batch_density_features(tiles.data(),ao.data(),work.data(),features.data(),0,spins,jets,
                            work_jets,terms,2,&error);
  });
  for (const auto& tile : tiles)
    for (I spin=0;spin<spins;++spin)
      for (I point=0;point<I(tile.count);++point) {
        double reference[5]{};
        for (I mu=0;mu<I(tile.active);++mu) {
          const auto source=tile.ao_offset+point*tile.active+mu;
          const auto product=tile.work_offset+(spin*jets*tile.count+point)*tile.active+mu;
          reference[0]+=ao[source]*work[product];
          for (I jet=1;jet<4;++jet) {
            reference[jet]+=2*ao[source+jet*tile.count*tile.active]*work[product];
            reference[4]+=0.5*ao[source+jet*tile.count*tile.active]*work[product+jet*tile.count*tile.active];
          }
        }
        for (I term=0;term<terms;++term)
          close(features[tile.begin*spins*terms+(spin*terms+term)*tile.count+point],reference[term]);
      }
  launch({64,4,1},{128,1,1},[&] {
    batch_potential_panels(tiles.data(),ao.data(),coefficients.data(),weights.data(),work.data(),
                            0,spins,terms,work_jets,&error);
  });
  launch({6,1,8},{16,16,1},[&] {
    batch_local_potentials(tiles.data(),ao.data(),work.data(),local.data(),spins,work_jets,&error);
  });
  std::size_t selected_density=0,incumbent_density=0,selected_potential=0,incumbent_potential=0;
  for (const auto& tile : tiles) {
    const auto columns=(tile.active+15)/16,padded_columns=16*columns,padded_points=16*((tile.count+15)/16);
    selected_density+=spins*work_jets*tile.active*tile.count*padded_columns;
    incumbent_density+=spins*work_jets*padded_columns*padded_points*padded_columns;
    selected_potential+=spins*work_jets*tile.active*(tile.active+1)/2*padded_points;
    incumbent_potential+=spins*work_jets*columns*(columns+1)/2*16*16*padded_points;
  }
  assert(density_terms.load()==selected_density && selected_density<incumbent_density);
  assert(potential_terms.load()==selected_potential && selected_potential<incumbent_potential);
  double totals[3]{},reference_totals[3]{};
  launch({2,1,1},{128,1,1},[&] {
    batch_ordered_scatter(tiles.data(),4,local.data(),point_totals.data(),0,ids.data(),full_n,
                          potential.data(),totals,&error);
  });
  // Explicit unfactored AO bilinears do not use packed work or local matrices.
  for (const auto& tile : tiles) {
    for (I spin=0;spin<spins;++spin)
      for (I mu=0;mu<I(tile.active);++mu)
        for (I nu=mu;nu<I(tile.active);++nu) {
          double value=0.0;
          for (I point=0;point<I(tile.count);++point) {
            const auto left=tile.ao_offset+point*tile.active+mu;
            const auto right=tile.ao_offset+point*tile.active+nu;
            const auto coefficient=tile.begin*spins*terms+spin*terms*tile.count+point;
            double integrand=coefficients[coefficient]*ao[left]*ao[right];
            for (I jet=1;jet<4;++jet) {
              const auto offset=jet*tile.count*tile.active;
              integrand+=coefficients[coefficient+jet*tile.count]*
                  (ao[left+offset]*ao[right]+ao[left]*ao[right+offset]);
              integrand+=coefficients[coefficient+4*tile.count]*ao[left+offset]*ao[right+offset];
            }
            value+=weights[tile.begin+point]*integrand;
          }
          const auto row=ids[tile.map_offset+mu],col=ids[tile.map_offset+nu];
          expected[(spin*full_n+row)*full_n+col]+=value;
          expected[(spin*full_n+col)*full_n+row]=expected[(spin*full_n+row)*full_n+col];
        }
    for (I channel=0;channel<3;++channel) {
      double value=0.0;
      for (I point=0;point<I(tile.count);++point)
        value+=point_totals[3*tile.begin+channel*tile.count+point];
      reference_totals[channel]+=value;
    }
  }
  for (std::size_t index=0;index<potential.size();++index) close(potential[index],expected[index]);
  for (I channel=0;channel<3;++channel) assert(totals[channel]==reference_totals[channel]);
  assert(error==0);
}
"""
