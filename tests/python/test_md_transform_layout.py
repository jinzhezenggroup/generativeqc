"""Compile structural-support planning and check its public expansion contract."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from _cpp_source_support import cpp_function_definition
from test_coulomb_optional_allocation import compile_cached_probe

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_md_transform_support_and_slot_bounds(tmp_path: Path) -> None:
    """Every possible axis derivative survives, including spherical unions."""
    source = tmp_path / "support.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <set>
#include "scf/md_transform_layout.hpp"
using namespace generativeqc::scf;

MdTransformShell cartesian(unsigned angular) {
  MdTransformShell shell;
  for(unsigned first=0;first<=angular;++first)
    for(unsigned second=0;second<=angular-first;++second)
      shell.push_back({{static_cast<std::uint8_t>(first),static_cast<std::uint8_t>(second),
                        static_cast<std::uint8_t>(angular-first-second)}});
  return shell;
}

void check(const MdTransformShell& first,const MdTransformShell& second,unsigned angular) {
  const auto layout=md_transform_layout(first,second,angular);
  const unsigned hermites=(angular+1)*(angular+2)*(angular+3)/6;
  assert(layout.slots.size()==first.size()*second.size()*hermites);
  std::size_t component=0;
  std::set<std::uint16_t> retained;
  for(const auto& bra:first) for(const auto& ket:second) {
    std::vector<bool> expected(hermites,false);
    for(const auto& bra_term:bra) for(const auto& ket_term:ket) {
      for(unsigned axis_x=0;axis_x<=bra_term[0]+ket_term[0];++axis_x)
        for(unsigned axis_y=0;axis_y<=bra_term[1]+ket_term[1];++axis_y)
          for(unsigned axis_z=0;axis_z<=bra_term[2]+ket_term[2];++axis_z) {
            unsigned index=0;
            for(unsigned preceding=0;preceding<axis_x;++preceding)
              index+=(angular-preceding+1)*(angular-preceding+2)/2;
            index+=axis_y*(angular-axis_x+1)-axis_y*(axis_y-1)/2+axis_z;
            expected.at(index)=true;
          }
    }
    for(unsigned derivative=0;derivative<hermites;++derivative) {
      const auto slot=layout.slots[component*hermites+derivative];
      assert(bool(slot)==expected[derivative]);
      assert(slot<layout.values);
      if(slot) assert(retained.insert(slot).second);
    }
    ++component;
  }
  assert(retained.size()+1==layout.values);
}

int main() {
  for(unsigned bra=0;bra<=2;++bra) for(unsigned ket=0;ket<=2;++ket)
    check(cartesian(bra),cartesian(ket),bra+ket);
  MdTransformShell spherical{{{1,1,0}},{{0,1,1}},{{0,0,2},{2,0,0},{0,2,0}},
                              {{1,0,1}},{{2,0,0},{0,2,0}}};
  for(unsigned angular=0;angular<=2;++angular) {
    check(cartesian(angular),spherical,angular+2);
    check(spherical,cartesian(angular),angular+2);
  }
  check(spherical,spherical,4);
  std::reverse(spherical.begin(),spherical.end());
  check(spherical,spherical,4);
  try { md_transform_layout(spherical,spherical,5); assert(false); }
  catch(const std::invalid_argument&) {}
}
""".replace("#include <cassert>", "#include <cassert>\n#include <algorithm>")
    )
    binary = tmp_path / "support"
    compile_cached_probe(source, binary)
    subprocess.run([str(binary)], check=True, timeout=10)


def test_sparse_preparation_against_polynomial_oracle(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Execute production preparation with host indices, not a GPU substitute.

    An independent long-double polynomial-to-Gaussian-derivative expansion
    checks all retained coefficients. This complements, but never replaces,
    resident GPU/Libcint, sanitizer and complete-endpoint qualification.
    """
    (tmp_path / "cuda_runtime_api.h").write_text(
        "#pragma once\nusing cudaStream_t=void*; using cudaError_t=int;\n"
    )
    kernel = cpp_function_definition(
        (ROOT / "src/scf/cuda/direct_md_j.cu").read_text(), "md_j_transforms"
    )
    source, binary = tmp_path / "numerical.cpp", tmp_path / "numerical"
    source.write_text(
        r"""
#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <limits>
#include <vector>
#define __host__
#define __device__
#define __global__
#include "scf/md_transform_layout.hpp"
#include "scf/cuda/direct_md_j.hpp"
#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/hermite_recurrence.cuh"
#include "scf/cuda/md_hermite_index.cuh"
using namespace generativeqc::scf;
using namespace generativeqc::scf::cuda_execution;
struct Index { unsigned x; } blockIdx{0},threadIdx{0},blockDim{32};
KERNEL

using Polynomial=std::vector<long double>;
long double oracle_axis(unsigned first,unsigned second,unsigned derivative,
                        double alpha,double beta,double center_first,double center_second) {
  const long double exponent=static_cast<long double>(alpha)+beta;
  const long double product=(alpha*static_cast<long double>(center_first)+
                             beta*static_cast<long double>(center_second))/exponent;
  Polynomial polynomial(first+second+1,0.0L);
  polynomial[0]=1.0L;
  unsigned degree=0;
  for(unsigned term=0;term<first+second;++term) {
    const long double displacement=product-(term<first ? center_first : center_second);
    Polynomial next(polynomial.size(),0.0L);
    for(unsigned power=0;power<=degree;++power) {
      next[power]+=displacement*polynomial[power];
      next[power+1]+=polynomial[power];
    }
    polynomial=std::move(next);
    ++degree;
  }
  std::vector<Polynomial> derivatives(first+second+1,Polynomial(first+second+1,0.0L));
  derivatives[0][0]=1.0L;
  for(unsigned order=1;order<=first+second;++order)
    for(unsigned power=0;power<order;++power) {
      derivatives[order][power+1]+=2*exponent*derivatives[order-1][power];
      if(power) derivatives[order][power-1]-=power*derivatives[order-1][power];
    }
  long double answer=0.0L;
  for(int order=static_cast<int>(first+second);order>=0;--order) {
    const long double coefficient=polynomial[order]/derivatives[order][order];
    if(static_cast<unsigned>(order)==derivative) answer=coefficient;
    for(int power=0;power<=order;++power)
      polynomial[power]-=coefficient*derivatives[order][power];
  }
  const long double distance=static_cast<long double>(center_first)-center_second;
  return answer*std::exp(-alpha*static_cast<long double>(beta)/exponent*distance*distance);
}

MdTransformShell shell(unsigned angular,bool spherical) {
  if(angular==2 && spherical)
    return {{{1,1,0}},{{0,1,1}},{{0,0,2},{2,0,0},{0,2,0}},
             {{1,0,1}},{{2,0,0},{0,2,0}}};
  MdTransformShell result;
  for(unsigned first=0;first<=angular;++first)
    for(unsigned second=0;second<=angular-first;++second)
      result.push_back({{static_cast<std::uint8_t>(first),static_cast<std::uint8_t>(second),
                         static_cast<std::uint8_t>(angular-first-second)}});
  return result;
}

void exercise(unsigned first,unsigned second,bool spherical,bool coincident) {
  const auto bra=shell(first,spherical),ket=shell(second,spherical);
  const auto layout=md_transform_layout(bra,ket,first+second);
  const auto hermites=md_j_hermite_count(first+second);
  MdJPair pair{0,0,1,0,static_cast<int>(bra.size()),static_cast<int>(bra.size()),
               static_cast<int>(ket.size()),first+second,hermites,0,1,0};
  MdJPrimitive primitive{0,0,1,0,0,0.0,{}};
  std::vector<double> values(layout.values,std::numeric_limits<double>::quiet_NaN());
  MdJView md;
  md.pairs=&pair; md.primitives=&primitive;
  md.transform_slots=layout.slots.data(); md.transforms=values.data();
  std::vector<std::uint8_t> counts,angular;
  std::vector<double> coefficients;
  for(const auto& expansion:{bra,ket}) for(const auto& component:expansion) {
    counts.push_back(component.size());
    for(unsigned term=0;term<3;++term) {
      for(unsigned axis=0;axis<3;++axis)
        angular.push_back(term<component.size() ? component[term][axis] : 0);
      coefficients.push_back(term<component.size() ? (term==1 ? -0.37 : 0.71) : 0.0);
    }
  }
  const std::int32_t atoms[2]={0,1};
  const std::uint8_t shell_angular[2]={static_cast<std::uint8_t>(first),
                                      static_cast<std::uint8_t>(second)};
  const double positions[6]={0.17,-0.23,0.38,coincident ? 0.17 : -0.21,
                             coincident ? -0.23 : 0.53,coincident ? 0.38 : 0.12};
  const double exponents[2]={0.713,1.932};
  double weights[2]={-0.31,0.87};
  DeviceBatch batch{};
  batch.nbf=counts.size(); batch.primitive_exponents=exponents;
  batch.primitive_coefficients=weights; batch.positions=positions;
  batch.shell_atoms=atoms; batch.shell_angular=shell_angular;
  batch.ao_term_counts=counts.data(); batch.ao_term_angular=angular.data();
  batch.ao_term_coefficients=coefficients.data();
  for(threadIdx.x=0;threadIdx.x<blockDim.x;++threadIdx.x) md_j_transforms(batch,md);
  std::size_t component=0;
  for(std::size_t bra_ao=0;bra_ao<bra.size();++bra_ao)
    for(std::size_t ket_ao=0;ket_ao<ket.size();++ket_ao) {
      for(unsigned derivative=0;derivative<hermites;++derivative) {
        const auto powers=md_hermite_angular(first+second,derivative);
        const unsigned axis_powers[3]={powers.x,powers.y,powers.z};
        long double expected=0.0L;
        for(std::size_t bra_term=0;bra_term<bra[bra_ao].size();++bra_term)
          for(std::size_t ket_term=0;ket_term<ket[ket_ao].size();++ket_term) {
            long double contribution=coefficients[bra_ao*3+bra_term]*
                static_cast<long double>(coefficients[(bra.size()+ket_ao)*3+ket_term]);
            for(unsigned axis=0;axis<3;++axis)
              contribution*=oracle_axis(bra[bra_ao][bra_term][axis],ket[ket_ao][ket_term][axis],
                                         axis_powers[axis],exponents[0],exponents[1],
                                         positions[axis],positions[axis+3]);
            expected+=contribution;
          }
        expected*=static_cast<long double>(weights[0])*weights[1];
        const double actual=md_j_transform(md,primitive,pair,component,derivative);
        assert(std::isfinite(actual));
        assert(std::abs(actual-expected)<2e-14L*(1+std::abs(expected)));
        if(layout.slots[component*hermites+derivative]==0)
          assert(actual==0.0 && std::signbit(actual));
      }
      ++component;
    }
  weights[0]=std::numeric_limits<double>::infinity();
  for(threadIdx.x=0;threadIdx.x<blockDim.x;++threadIdx.x) md_j_transforms(batch,md);
  assert(std::isnan(values[0]));
  for(unsigned slot=0;slot<layout.slots.size();++slot)
    if(!layout.slots[slot])
      assert(std::isnan(md_j_transform(md,primitive,pair,slot/hermites,slot%hermites)));
}

int main() {
  for(unsigned first=0;first<=2;++first) for(unsigned second=0;second<=2;++second)
    for(bool spherical:{false,true}) for(bool coincident:{false,true})
      exercise(first,second,spherical,coincident);
}
""".replace("KERNEL", kernel)
    )
    native_cxx.build_executable(
        [source],
        binary,
        compile_args=[
            "-std=c++20",
            f"-I{tmp_path}",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ],
    )
    subprocess.run([str(binary)], check=True, timeout=10)
