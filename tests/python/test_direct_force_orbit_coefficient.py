"""Independently enumerate ERI orbits for the guarded Direct-force contraction."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from generativeqc_compiler.integral.lowering.fock_accumulation import (
    emit_direct_force_density_coefficient,
)

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import NativeCxx


def test_restricted_orbits_and_retained_general_inputs(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Check every canonical coincidence pattern, signed coefficients and fallbacks.

    The oracle explicitly deduplicates all eight index tuples and accumulates
    in long double. It does not reuse the production stabilizer formula.
    """
    generated = (
        emit_direct_force_density_coefficient()
        .replace("__device__ ", "")
        .replace("__forceinline__ ", "")
    )
    source = tmp_path / "orbits.cpp"
    source.write_text(PREFIX + generated + PROBE)
    executable = tmp_path / "orbits"
    native_cxx.build_executable(
        [source],
        executable,
        compile_args=("-std=c++20", "-O2", "-ffp-contract=off"),
        compile_timeout=60,
        link_timeout=30,
    )
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)


PREFIX = r"""
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>

using Quartet = std::array<std::size_t, 4>;
using Matrix = std::array<double, 50>;
std::size_t matrix_index(std::size_t i, std::size_t j, std::size_t n) { return i*n+j; }
Quartet permute(unsigned p, std::size_t i, std::size_t j, std::size_t k, std::size_t l) {
  const Quartet orbit[] = {{i,j,k,l},{j,i,k,l},{i,j,l,k},{j,i,l,k},
                           {k,l,i,j},{l,k,i,j},{k,l,j,i},{l,k,j,i}};
  return orbit[p];
}
void eri_symmetry_permutation(unsigned p, std::size_t i, std::size_t j,
                             std::size_t k, std::size_t l,
                             std::size_t& a, std::size_t& b,
                             std::size_t& c, std::size_t& d) {
  const auto q = permute(p,i,j,k,l); a=q[0]; b=q[1]; c=q[2]; d=q[3];
}
bool unique_eri_symmetry_permutation(unsigned p, std::size_t i, std::size_t j,
                                    std::size_t k, std::size_t l) {
  const auto q = permute(p,i,j,k,l);
  for (unsigned previous=0; previous<p; ++previous)
    if (permute(previous,i,j,k,l)==q) return false;
  return true;
}
"""

PROBE = r"""
template<class Float>
Float oracle(const Matrix& density, const Matrix* seed, Quartet q, double j, double k,
             bool unrestricted=false) {
  std::set<Quartet> visited;
  Float value=0;
  auto product=[&](std::size_t a, std::size_t b, Float scale) {
    if (seed)
      return scale*Float(density[a])*Float((*seed)[b])+
             scale*Float((*seed)[a])*Float(density[b]);
    return scale*Float(density[a])*Float(density[b]);
  };
  for (unsigned p=0; p<8; ++p) {
    const auto t=permute(p,q[0],q[1],q[2],q[3]);
    if (!visited.insert(t).second) continue;
    const auto ab=t[0]*5+t[1], cd=t[2]*5+t[3], ac=t[0]*5+t[2], bd=t[1]*5+t[3];
    if (j!=0) {
      if (unrestricted)
        value+=Float(0.5)*Float(j)*(Float(density[ab])+Float(density[25+ab]))*
                                  (Float(density[cd])+Float(density[25+cd]));
      else value+=product(ab,cd,Float(0.5)*Float(j));
    }
    if (k!=0) {
      if (unrestricted)
        value+=Float(0.5)*Float(k)*(Float(density[ac])*Float(density[bd])+
                                  Float(density[25+ac])*Float(density[25+bd]));
      else value+=product(ac,bd,Float(0.5)*Float(k));
    }
  }
  return value;
}
bool close(double actual,long double expected) {
  return std::isfinite(actual) &&
         std::abs(static_cast<long double>(actual)-expected)<=
             4e-14L*std::max(1.L,std::abs(expected));
}
int main() {
  Matrix d{},seed{};
  for (std::size_t spin=0;spin<2;++spin)
    for (std::size_t i=0;i<5;++i)
      for (std::size_t j=0;j<5;++j) {
        d[spin*25+i*5+j]=std::sin(double((i+j)*3+i*j+spin)*0.17);
        seed[spin*25+i*5+j]=std::cos(double(i*3+j+spin)*0.21);
      }
  for (double j:{0.,1.,-0.7}) for (double k:{0.,-0.125,-0.5,0.38})
    for (std::size_t i=0;i<5;++i) for (std::size_t b=0;b<=i;++b)
      for (std::size_t c=0;c<5;++c) for (std::size_t e=0;e<=c;++e) {
        const Quartet q{i,b,c,e};
        double fast{};
        if (!direct_force_restricted_orbit_coefficient(5,0,d.data(),i,b,c,e,j,k,fast)) return 1;
        if (!close(fast,oracle<long double>(d,nullptr,q,j,k))) return 2;
        const auto actual=direct_force_density_coefficient_scaled<false>(
            5,0,0,d.data(),i,b,c,e,j,k);
        if (!close(actual,oracle<long double>(d,nullptr,q,j,k))) return 3;
        const auto directional=direct_force_density_coefficient_scaled<false>(
            5,0,0,d.data(),i,b,c,e,j,k,seed.data());
        if (!close(directional,oracle<long double>(d,&seed,q,j,k))) return 4;
        const auto unrestricted=direct_force_density_coefficient_scaled<true>(
            5,0,0,d.data(),i,b,c,e,j,k);
        if (!close(unrestricted,oracle<long double>(d,nullptr,q,j,k,true))) return 5;
      }
  // Batched restricted owners publish a matrix-relative physical offset.
  Matrix shifted{};
  std::copy_n(d.begin()+25,25,shifted.begin());
  const auto offset_value=direct_force_density_coefficient_scaled<false>(
      5,25,0,d.data(),4,3,2,1,0.8,-0.17);
  if (!close(offset_value,oracle<long double>(shifted,nullptr,Quartet{4,3,2,1},0.8,-0.17))) return 11;
  // Every used asymmetric pair must keep the general permutation semantics.
  for (std::size_t i=0;i<25;++i) d[i]+=0.013*double(i);
  const Quartet q{4,3,2,1};
  double ignored{};
  if (direct_force_restricted_orbit_coefficient(5,0,d.data(),4,3,2,1,1,-0.125,ignored)) return 6;
  auto actual=direct_force_density_coefficient_scaled<false>(5,0,0,d.data(),4,3,2,1,1,-0.125);
  const auto expected=oracle<double>(d,nullptr,q,1,-0.125);
  if (std::bit_cast<std::uint64_t>(actual)!=std::bit_cast<std::uint64_t>(expected)) return 7;
  // Extreme and subnormal factors use the existing scale-before-product order.
  for (const auto factors:{std::array{1e200,1e-200},std::array{1e-200,1e200},
                           std::array{std::numeric_limits<double>::denorm_min(),1e100}}) {
    d.fill(factors[0]);
    if (direct_force_restricted_orbit_coefficient(5,0,d.data(),4,3,2,1,factors[1],0,ignored)) return 8;
    actual=direct_force_density_coefficient_scaled<false>(5,0,0,d.data(),4,3,2,1,factors[1],0);
    const auto reference=oracle<double>(d,nullptr,q,factors[1],0);
    if (std::bit_cast<std::uint64_t>(actual)!=std::bit_cast<std::uint64_t>(reference)) return 9;
  }
  // A disabled source must not be made active by an exceptional unused matrix.
  d.fill(std::numeric_limits<double>::infinity());
  if (!direct_force_restricted_orbit_coefficient(5,0,d.data(),4,3,2,1,0,0,ignored) || ignored!=0) return 10;
}
"""
