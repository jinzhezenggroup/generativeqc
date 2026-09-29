"""Compare the exact split-response planner against its still-live owners."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _fragment(source: str) -> str:
    start = source.index("  const auto small_response_retained =")
    return source[start : source.index("  const auto coordinates =", start)]


def _probe(fragment: str) -> str:
    return (
        r'''
#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>
std::size_t hs=1,he=1,fs=1,oj=1;
std::size_t checked_add(std::size_t a,std::size_t b) {
  if(b>std::numeric_limits<std::size_t>::max()-a) throw std::length_error("overflow");
  return a+b;
}
std::size_t checked_mul(std::size_t a,std::size_t b) {
  if(a && b>std::numeric_limits<std::size_t>::max()/a) throw std::length_error("overflow");
  return a*b;
}
std::size_t bytes(std::size_t n) {return checked_mul(n,sizeof(double));}
std::size_t square(std::size_t n) {return checked_mul(n,n);}
std::size_t sum(std::initializer_list<std::size_t> values) {
  std::size_t out=0; for(auto v:values) out=checked_add(out,v); return out;
}
namespace generated {
std::size_t hamiltonian_small_weights_arena_elements(std::size_t,std::size_t) {return hs;}
std::size_t hamiltonian_eri_weights_arena_elements(std::size_t,std::size_t) {return he;}
std::size_t fock_small_weights_arena_elements(std::size_t,std::size_t) {return fs;}
std::size_t orbital_jvp_arena_elements(std::size_t,std::size_t) {return oj;}
}
namespace response {
struct GmresOptions {std::size_t restart{},max_workspace_bytes{};};
struct Plan {std::size_t workspace_bytes{};};
Plan prepare_gmres(std::size_t n,GmresOptions) {return {64*n+128};}
}
int main() {
  for(std::size_t o=1;o<=5;++o) for(std::size_t v=1;v<=7;++v) {
    const auto n=o+v,n2=n*n,n4=n2*n2,ov=o*v;
    const std::size_t before_raw=12345,raw_retained=8*(n4+3*n2);
    const auto max_bytes=std::numeric_limits<std::size_t>::max();
    for(unsigned dominant=0;dominant<4;++dominant) {
      hs=he=fs=oj=1;
      if(dominant==0) hs=100000;
      if(dominant==1) he=100000;
      if(dominant==2) fs=100000;
      if(dominant==3) oj=100000;
      struct {std::size_t response_phase_bytes{};} plan;
'''
        + fragment
        + r'''
      // Independent inventory at the final calls: correlation/canonicalization,
      // two Fock seeds, d_rotation, the retained orbital arena, dense response
      // matrix, basis/action vectors, Z solution, and independent residual.
      const auto input=before_raw+raw_retained;
      const auto compact=8*(4*n2+ov);
      const auto final_inputs=input+2*compact+8*(2*n2+n2+oj+ov*ov+4*ov);
      const auto derivative=final_inputs+compact+8*n4;
      if(derivative_live<derivative) {
        std::cerr<<"final derivative omits live response owners\n"; return 1;
      }
      if(plan.response_phase_bytes<final_inputs+compact+8*hs ||
         plan.response_phase_bytes<final_inputs+compact+8*n4+8*he ||
         plan.response_phase_bytes<input+2*compact+8*(2*n2+fs)) {
        std::cerr<<"response peak omits a live owner\n"; return 2;
      }
    }
  }
  std::cout<<"140 split-response live-owner budget cases passed\n";
}
'''
    )


def test_split_response_budget_covers_final_live_owners(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    text = (ROOT / "src/cc/rccsdt_force.cpp").read_text()
    source = tmp_path / "budget.cpp"
    source.write_text(_probe(_fragment(text)))
    executable = tmp_path / "budget"
    subprocess.run(
        [compiler, "-std=c++20", "-O0", str(source), "-o", str(executable)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stdout + result.stderr
