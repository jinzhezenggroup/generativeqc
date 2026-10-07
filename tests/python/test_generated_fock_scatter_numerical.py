"""Generated Direct-Fock scatter matches an independent dense ERI contraction."""

import subprocess
from pathlib import Path

from generativeqc_compiler.integral.lowering.fock_accumulation import (
    emit_direct_fock_accumulation_header,
    emit_generated_shell_fock_accumulation,
)

ROOT = Path(__file__).resolve().parents[2]


def test_restricted_raw_k_block_mapping_matches_eightfold_scatter() -> None:
    """The shell-block K map must preserve every canonical symmetry channel."""

    n = 4
    density = [[0.031 * (1 + row + 3 * column) for column in range(n)] for row in range(n)]

    def permutation(
        p: int, i: int, j: int, k: int, l: int
    ) -> tuple[int, int, int, int]:
        values = (
            (i, j, k, l),
            (j, i, k, l),
            (i, j, l, k),
            (j, i, l, k),
            (k, l, i, j),
            (l, k, i, j),
            (k, l, j, i),
            (l, k, j, i),
        )
        return values[p]

    def unique(p: int, i: int, j: int, k: int, l: int) -> bool:
        pair_swapped = p >= 4
        first_pair_diagonal = k == l if pair_swapped else i == j
        second_pair_diagonal = i == j if pair_swapped else k == l
        if p & 1 and first_pair_diagonal:
            return False
        if p & 2 and second_pair_diagonal:
            return False
        return not pair_swapped or i != k or j != l

    def add(
        target: dict[tuple[int, int], float], row: int, column: int, value: float
    ) -> None:
        target[(row, column)] = target.get((row, column), 0.0) + value

    cases = 0
    for i in range(n):
        for j in range(i + 1):
            first_pair = i * (i + 1) // 2 + j
            for k in range(n):
                for l in range(k + 1):
                    second_pair = k * (k + 1) // 2 + l
                    if first_pair < second_pair:
                        continue
                    integral = 0.127 + 0.011 * (i + 2 * j + 3 * k + 5 * l)

                    incumbent: dict[tuple[int, int], float] = {}
                    for p in range(8):
                        if not unique(p, i, j, k, l):
                            continue
                        a, b, c, d = permutation(p, i, j, k, l)
                        add(incumbent, a, c, density[b][d] * integral)

                    blocked: dict[tuple[int, int], float] = {}
                    add(blocked, i, k, density[j][l] * integral)
                    if i != j:
                        add(blocked, j, k, density[i][l] * integral)
                    if k != l:
                        add(blocked, i, l, density[j][k] * integral)
                    if i != j and k != l:
                        add(blocked, j, l, density[i][k] * integral)
                    if i != k or j != l:
                        add(blocked, k, i, density[l][j] * integral)
                        if k != l:
                            add(blocked, l, i, density[k][j] * integral)
                        if i != j:
                            add(blocked, k, j, density[l][i] * integral)
                        if i != j and k != l:
                            add(blocked, l, j, density[k][i] * integral)

                    assert blocked.keys() == incumbent.keys()
                    for key, expected in incumbent.items():
                        assert blocked[key] == pytest.approx(expected, abs=1.0e-15)
                    cases += 1

    assert cases == 55


def test_generated_scatter_all_canonical_index_coincidences(
    tmp_path: Path, native_cxx: object
) -> None:
    (tmp_path / "cuda_runtime.h").write_text(
        "#pragma once\n#define __device__\n#define __host__\n#define __forceinline__ inline\ninline void atomicAdd(double* p,double x){*p+=x;}\n"
    )
    (tmp_path / "scatter.hpp").write_text(emit_direct_fock_accumulation_header())
    (tmp_path / "shell_scatter.hpp").write_text(
        r"""
using namespace generativeqc::scf::cuda_execution;
struct GeneratedDpppShellTask {
  std::size_t matrix_order, density_offset, spin_offset;
  unsigned reversed_shell_pair_mask;
};
constexpr unsigned kGeneratedDpppCoulombConsumerBit=4U;
constexpr unsigned kGeneratedDpppExchangeConsumerBit=8U;
std::size_t generated_dppp_matrix_index(std::size_t a, std::size_t b, std::size_t n) {
  return a+b*n;
}
void generated_dppp_eri_permutation(unsigned p, std::size_t i, std::size_t j,
    std::size_t k, std::size_t l, std::size_t& a, std::size_t& b,
    std::size_t& c, std::size_t& d) {
  eri_symmetry_permutation(p,i,j,k,l,a,b,c,d);
}
bool generated_dppp_unique_permutation(unsigned p, std::size_t i, std::size_t j,
    std::size_t k, std::size_t l, std::size_t, std::size_t, std::size_t, std::size_t) {
  return unique_eri_symmetry_permutation(p,i,j,k,l);
}
"""
        + emit_generated_shell_fock_accumulation()
    )
    driver = tmp_path / "scatter.cpp"
    driver.write_text(
        r"""
#include "scatter.hpp"
#include "shell_scatter.hpp"
#include <set>
#include <vector>
#include <iostream>
#include <array>
#include <cmath>
int main() {
  using namespace generativeqc::scf::cuda_execution;
  constexpr size_t n=4, m=n*n, off=5, spin=40;
  auto index=[](size_t p,size_t q,size_t r,size_t s){return ((p*n+q)*n+r)*n+s;};
"""
        + r"""
  size_t cases=0;
  for(size_t i=0;i<n;++i) for(size_t j=0;j<=i;++j)
  for(size_t k=0;k<n;++k) for(size_t l=0;l<=k;++l) {
    if(i*(i+1)/2+j < k*(k+1)/2+l) continue;
    const double value=.125+.003*(i+j+k+l);
    std::set<std::array<size_t,4>> orbit{{i,j,k,l},{j,i,k,l},{i,j,l,k},{j,i,l,k},
                                     {k,l,i,j},{l,k,i,j},{k,l,j,i},{l,k,j,i}};
    std::vector<double> eri(n*n*n*n), density(100), seed(100);
    for(auto a:orbit) eri[index(a[0],a[1],a[2],a[3])]=value;
    for(size_t p=0;p<n;++p) for(size_t q=0;q<n;++q) {
      density[off+p+q*n]=((p+q)%3==0 ? 0.0 : .03*(p+2*q+1));
      density[spin+p+q*n]=.04*(p+2*q+1);
      density[spin+m+p+q*n]=-.01*(2*p+q+2);
      seed[off+p*n+q]=.02*(int(p)-2*int(q)+1);
    }
    // Dense ordered contraction checks the bilinear coefficient for every
    // repeated-index orbit, including signed and nonsymmetric operands.
    double bilinear=0;
    for(size_t p=0;p<n;++p) for(size_t q=0;q<n;++q)
    for(size_t r=0;r<n;++r) for(size_t s=0;s<n;++s)
      bilinear+=seed[off+p*n+q]*density[off+r*n+s]*
                  (eri[index(p,q,r,s)]-.5*eri[index(p,r,q,s)]);
    const double coefficient=direct_bilinear_density_coefficient(
        n,off,density.data(),seed.data(),i,j,k,l);
    if(std::abs(value*coefficient-bilinear)>2e-13) return 7;
    for(bool unrestricted : {false,true}) for(unsigned mode : {0U,1U,2U,3U}) {
      const bool coulomb_only=mode==1U, exchange_only=mode>=2U, hf_exchange=mode==3U;
      std::vector<double> actual(100), shell_actual(100), expected(100);
      const GeneratedDpppShellTask task{n,off,spin,mode*4U};
      if(unrestricted) accumulate_direct_fock_integral<true>(
          n,off,spin,density.data(),actual.data(),i,j,k,l,value,coulomb_only,exchange_only,hf_exchange);
      else accumulate_direct_fock_integral<false>(
          n,off,spin,density.data(),actual.data(),i,j,k,l,value,coulomb_only,exchange_only,hf_exchange);
      if(unrestricted) generated_dppp_accumulate_fock<true>(
          task,density.data(),shell_actual.data(),i,j,k,l,value);
      else generated_dppp_accumulate_fock<false>(
          task,density.data(),shell_actual.data(),i,j,k,l,value);
      for(size_t p=0;p<n;++p) for(size_t q=0;q<n;++q)
      for(size_t r=0;r<n;++r) for(size_t s=0;s<n;++s) {
        const double J=eri[index(p,q,r,s)], K=eri[index(p,r,q,s)];
        if(unrestricted) {
          const double a=density[spin+r+s*n], b=density[spin+m+r+s*n];
          if(exchange_only) {
            expected[spin+p+q*n]+=(hf_exchange?-1.0:1.0)*a*K;
            expected[spin+m+p+q*n]+=(hf_exchange?-1.0:1.0)*b*K;
          } else {
            expected[spin+p+q*n]+=(a+b)*J-(coulomb_only?0.0:a*K);
            expected[spin+m+p+q*n]+=(a+b)*J-(coulomb_only?0.0:b*K);
          }
        } else if(exchange_only) {
          expected[off+p+q*n]+=(hf_exchange?-.5:1.0)*density[off+r+s*n]*K;
        } else {
          expected[off+p+q*n]+=density[off+r+s*n]*(J-(coulomb_only?0.0:.5*K));
        }
      }
      for(size_t a=0;a<actual.size();++a) if(std::abs(actual[a]-expected[a])>2e-13) return 1;
      for(size_t a=0;a<shell_actual.size();++a) if(std::abs(shell_actual[a]-expected[a])>2e-13) return 8;
      ++cases;
    }
  }
  if(cases!=440) return 2;
  {
    constexpr float mixed_value = 0.9876543F;
    std::vector<double> density(100), actual(100);
    density[off] = 0.123456789123;
    accumulate_direct_fock_integral<false, true>(
        1, off, spin, density.data(), actual.data(), 0, 0, 0, 0,
        mixed_value, true);
    const double expected = static_cast<double>(
        static_cast<float>(density[off]) * mixed_value);
    if(std::abs(actual[off]-expected)>1e-15) return 3;
    if(std::abs(actual[off]-density[off]*static_cast<double>(mixed_value))<1e-10) return 4;
  }
  {
    std::vector<double> density(100), actual(100);
    density[off + 1 + 1*n] = 1e308;
    density[off + 1 + 3*n] = 1e308;
    density[off + 3 + 1*n] = 1e308;
    density[off + 3 + 3*n] = 1e308;
    accumulate_direct_fock_integral<false>(
        n, off, spin, density.data(), actual.data(), 3, 2, 1, 0,
        2.0, false);
    bool saw_finite_exchange = false;
    for(double value : actual) {
      if(!std::isfinite(value)) return 5;
      if(value != 0.0) saw_finite_exchange = true;
    }
    if(!saw_finite_exchange) return 6;
  }
  std::cout<<cases<<" independent dense RHF/UHF scatter comparisons passed\n";
}
"""
    )
    executable = tmp_path / "scatter"
    native_cxx.build_executable(
        [driver],
        executable,
        compile_args=(
            "-std=c++20",
            "-O2",
            "-I" + str(tmp_path),
            "-I" + str(ROOT / "src"),
        ),
        compile_timeout=60,
        link_timeout=60,
    )
    result = subprocess.run(
        [str(executable)], check=False, capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stdout + result.stderr
