"""Host-only adapter composing actual production MD-J preparation/publication.

Scalar CUDA index doubles execute only the noncollective production kernels.
The canonical source loop is test orchestration; real threaded source scheduling
is qualified separately. No code here belongs to the production runtime.
"""

from __future__ import annotations

import ctypes as ct
from pathlib import Path
from typing import Any

from _cpp_source_support import cpp_function_definition

ROOT = Path(__file__).resolve().parents[2]

PREAMBLE = r"""
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>
#define __host__
#define __device__
#define __global__
#define __constant__
#define __forceinline__ inline
#include "generated_md_j_engine.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/hermite_recurrence.cuh"
#include "scf/cuda/md_hermite_index.cuh"
#include "scf/md_transform_layout.hpp"
using namespace generativeqc::scf::cuda_execution;
using namespace generativeqc::scf;
struct Dim { unsigned x; };
Dim threadIdx{}, blockIdx{};
constexpr Dim blockDim{32};
"""

DRIVER = r"""
extern "C" int pipeline_j(unsigned nshell,unsigned nbf,const double* positions,
    const std::int32_t* atoms,const std::uint8_t* angular,
    const std::int64_t* primitive_offsets,const std::int64_t* ao_offsets,
    const double* exponents,const double* coefficients,const std::uint8_t* ao_counts,
    const std::uint8_t* ao_angular,const double* ao_coefficients,
    const double* density,const double* beta,int unrestricted,double* output,
    double* metrics) {
  DeviceBatch batch{};
  batch.nbf=nbf; batch.positions=positions; batch.shell_atoms=atoms;
  batch.shell_angular=angular; batch.primitive_exponents=exponents;
  batch.primitive_coefficients=coefficients; batch.ao_term_counts=ao_counts;
  batch.ao_term_angular=ao_angular; batch.ao_term_coefficients=ao_coefficients;
  std::vector<MdTransformShell> shells(nshell);
  for(unsigned shell=0;shell<nshell;++shell)
    for(auto ao=ao_offsets[shell];ao<ao_offsets[shell+1];++ao) {
      MdTransformComponent terms;
      for(unsigned term=0;term<ao_counts[ao];++term)
        terms.push_back({ao_angular[ao*9+term*3],ao_angular[ao*9+term*3+1],
                         ao_angular[ao*9+term*3+2]});
      shells[shell].push_back(terms);
    }
  std::vector<MdJPair> pairs;
  std::vector<MdJPrimitive> primitives;
  std::vector<std::uint16_t> slots;
  std::size_t transform_count=0;
  for(unsigned first=0;first<nshell;++first) for(unsigned second=0;second<=first;++second) {
    const unsigned L=angular[first]+angular[second];
    const auto layout=md_transform_layout(shells[first],shells[second],L);
    MdJPair pair{};
    pair.first=first; pair.second=second; pair.first_ao=ao_offsets[first];
    pair.second_ao=ao_offsets[second]; pair.first_count=shells[first].size();
    pair.second_count=shells[second].size(); pair.angular=L;
    pair.hermites=md_j_hermite_count(L); pair.transform_map=slots.size();
    slots.insert(slots.end(),layout.slots.begin(),layout.slots.end());
    pair.primitive_begin=primitives.size();
    for(auto a=primitive_offsets[first];a<primitive_offsets[first+1];++a)
      for(auto b=primitive_offsets[second];b<primitive_offsets[second+1];++b) {
        // The production transform represents both exchanged primitives for a
        // same-shell off-diagonal pair. Feed it that triangular inventory;
        // independent libcint still determines the expected full contracted J.
        if(first==second && b>a) continue;
        MdJPrimitive primitive{};
        primitive.pair=pairs.size(); primitive.first=a; primitive.second=b;
        primitive.transform_offset=transform_count;
        primitives.push_back(primitive);
        transform_count+=layout.values;
      }
    pair.primitive_end=primitives.size(); pairs.push_back(pair);
  }
  std::vector<std::uint32_t> order(primitives.size());
  std::iota(order.begin(),order.end(),0);
  // Nontrivial within-class order proves that output still follows public AOs.
  std::stable_sort(order.begin(),order.end(),[&](auto a,auto b) {
    const auto la=pairs[primitives[a].pair].angular,lb=pairs[primitives[b].pair].angular;
    return la==lb ? a>b : la<lb;
  });
  MdJView md{};
  md.pairs=pairs.data(); md.primitives=primitives.data();
  md.pair_count=pairs.size(); md.primitive_count=primitives.size();
  md.ordered_primitives=order.data(); md.transform_slots=slots.data();
  std::size_t cursor=0,packed=0;
  for(unsigned L=0;L<5;++L) {
    md.class_offsets[L]=cursor; md.pair_loc_offsets[L]=packed;
    while(cursor<order.size() && pairs[primitives[order[cursor]].pair].angular==L) {
      ++cursor; packed+=md_j_hermite_count(L);
    }
  }
  md.class_offsets[5]=cursor; md.pair_loc_offsets[5]=packed;
  const double nan=std::numeric_limits<double>::quiet_NaN();
  std::vector<double> transforms(transform_count,nan),hermite_density(packed,nan),potential(packed,nan);
  md.transforms=transforms.data(); md.density=hermite_density.data(); md.potential=potential.data();
  for(unsigned L=0;L<5;++L) {
    const auto count=md.class_offsets[L+1]-md.class_offsets[L];
    for(blockIdx.x=0;blockIdx.x<(count+31)/32;++blockIdx.x)
      for(threadIdx.x=0;threadIdx.x<32;++threadIdx.x) md_j_engine_pair_loc(md,L);
  }
  for(blockIdx.x=0;blockIdx.x<primitives.size();++blockIdx.x)
    for(threadIdx.x=0;threadIdx.x<32;++threadIdx.x) md_j_transforms(batch,md);
  for(blockIdx.x=0;blockIdx.x<primitives.size();++blockIdx.x)
    for(threadIdx.x=0;threadIdx.x<32;++threadIdx.x)
      md_j_density(batch,md,0,1,unrestricted,density,beta);
  // One canonical radial source, both density consumers except at source self.
  for(std::size_t i=0;i<order.size();++i) for(std::size_t j=i;j<order.size();++j) {
    const auto& bra=primitives[order[i]]; const auto& ket=primitives[order[j]];
    const unsigned a=pairs[bra.pair].angular,b=pairs[ket.pair].angular;
    double forward[35]{},reverse[35]{};
    switch(5*a+b) {
      DISPATCH_CASES
      default:return 1;
    }
    for(unsigned k=0;k<md_j_hermite_count(a);++k) potential[bra.hermite_offset+k]+=forward[k];
    for(unsigned k=0;k<md_j_hermite_count(b);++k) potential[ket.hermite_offset+k]+=reverse[k];
  }
  std::fill(output,output+nbf*nbf,0.0);
  for(blockIdx.x=0;blockIdx.x<pairs.size();++blockIdx.x)
    for(threadIdx.x=0;threadIdx.x<32;++threadIdx.x) md_j_project(batch,md,0,1,output);
  metrics[0]=metrics[1]=metrics[2]=0.0;
  unsigned stage=0;
  for(const auto* values:{&transforms,&hermite_density,&potential}) {
    for(const auto value:*values) {
      if(!std::isfinite(value)) return 2;
      metrics[stage]=std::max(metrics[stage],std::abs(value));
    }
    ++stage;
  }
  return std::all_of(output,output+nbf*nbf,[](double value){return std::isfinite(value);}) ? 0 : 3;
}
"""


def build_pipeline(native_cxx: Any, directory: Path) -> Any:
    """Compile unmodified production bodies, not a copied transform algorithm."""
    from generativeqc_compiler.integral.md_j_engine_cuda import emit_md_j_engine_header

    (directory / "generated_md_j_engine.cuh").write_text(emit_md_j_engine_header())
    (directory / "cuda_runtime_api.h").write_text(
        "#pragma once\nusing cudaError_t=int; using cudaStream_t=void*;\n"
    )
    native = (ROOT / "src/scf/cuda/direct_md_j.cu").read_text()
    kernels = "\n".join(
        cpp_function_definition(native, name)
        for name in ("md_j_transforms", "md_j_density", "md_j_project")
    )
    kernels += cpp_function_definition(
        (ROOT / "src/scf/cuda/md_j_engine.cuh").read_text(), "md_j_engine_pair_loc"
    )
    cases = "\n".join(
        f"case {5 * a + b}: md_engine_contract<{a},{b}>(bra,ket,md,true,i!=j,forward,reverse); break;"
        for a in range(5)
        for b in range(a, 5)
    )
    source, binary = directory / "pipeline.cpp", directory / "pipeline.so"
    source.write_text(PREAMBLE + kernels + DRIVER.replace("DISPATCH_CASES", cases))
    native_cxx.build_shared(
        [source],
        binary,
        compile_args=[
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            f"-I{directory}",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ],
    )
    library = ct.CDLL(str(binary))
    library.pipeline_j.argtypes = [
        ct.c_uint,
        ct.c_uint,
        *([ct.c_void_p] * 12),
        ct.c_int,
        ct.c_void_p,
        ct.c_void_p,
    ]
    library.pipeline_j.restype = ct.c_int
    return library
