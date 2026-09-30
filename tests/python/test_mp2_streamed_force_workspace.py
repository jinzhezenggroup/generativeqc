"""Execute production MP2 workspace admission and rotation allocation lifetimes.

The provider fixture models a returned MO block plus per-call staging. The actual
rotation consumer and workspace helper are compiled from repository sources.
This does not substitute for a full native force or NVIDIA execution gate.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

PREFIX = r"""
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <vector>
namespace trace {
struct alignas(std::max_align_t) Header { std::size_t size; bool tracked; };
bool active=false;
std::size_t live=0,peak=0;
}
void* operator new(std::size_t n) {
  auto* h=static_cast<trace::Header*>(std::malloc(sizeof(trace::Header)+n));
  if(!h) throw std::bad_alloc();
  h->size=n;h->tracked=trace::active;
  if(h->tracked) {trace::live+=n;trace::peak=std::max(trace::peak,trace::live);}
  return h+1;
}
void operator delete(void* p) noexcept {
  if(!p) return;
  auto* h=static_cast<trace::Header*>(p)-1;
  if(h->tracked) trace::live-=h->size;
  std::free(h);
}
void operator delete(void* p,std::size_t) noexcept {::operator delete(p);}
namespace generativeqc::posthf {
// CHECKED_ARITHMETIC
using Shape=std::array<std::size_t,4>;
std::size_t output_bytes(Shape shape) {
  std::size_t out=8;for(auto n:shape) out=checked_mul(out,n);return out;
}
struct MOBlockProvider {
  std::size_t n;bool staged;
  std::size_t scratch(Shape shape,bool cuda) const {
    if(!staged) return 0;
    return (cuda ? 65536 : 4096)+8*(shape[0]+3*shape[1]+7*shape[2]+13*shape[3]);
  }
  std::size_t peak(Shape shape,bool cuda) const {
    return output_bytes(shape)+scratch(shape,cuda);
  }
  std::vector<double> get(const std::array<std::vector<std::size_t>,4>& slots,
                          bool cuda,int) const {
    Shape shape{};for(unsigned a=0;a<4;++a) shape[a]=slots[a].size();
    std::vector<unsigned char> work(scratch(shape,cuda));
    std::vector<double> out(output_bytes(shape)/8,0.0);
    return out;
  }
};
}
"""

ROTATION_PREFIX = r"""
namespace generativeqc::mp2 {
std::size_t square(std::size_t n) {return posthf::checked_mul(n,n);}
std::size_t g_index(std::size_t o,std::size_t v,std::size_t i,std::size_t j,
                    std::size_t a,std::size_t b) {return ((i*o+j)*v+a)*v+b;}
bool finite(std::span<const double> a) {
  return std::all_of(a.begin(),a.end(),[](double x){return std::isfinite(x);});
}
std::vector<std::size_t> all_orbitals(std::size_t n) {
  std::vector<std::size_t> out(n);for(std::size_t i=0;i<n;++i) out[i]=i;return out;
}
"""

MAIN = r"""
}
int main(int argc,char** argv) {
  if(argc!=5) return 1;
  const std::size_t n=std::strtoul(argv[1],nullptr,10),o=std::strtoul(argv[2],nullptr,10);
  const bool cuda=std::atoi(argv[3]),staged=std::atoi(argv[4]);
  using namespace generativeqc;
  posthf::MOBlockProvider provider{n,staged};
  std::vector<posthf::Shape> queried;
  const auto bound=mp2::detail::streamed_force_workspace_bytes(n,o,[&](auto shape) {
    queried.push_back(shape);return provider.peak(shape,cuda);
  });
  if(queried.size()!=10) return 2;
  // Independently enumerate all returned-block lifetimes, including a different
  // staging peak for every orientation and the two live adjoint input copies.
  const auto v=n-o,ov=o*v,n2=n*n,n3=n2*n;
  const std::array<posthf::Shape,10> shapes={{{o,v,o,v},{n,n,1,1},{n,1,1,n},
    {1,n,n,n},{n,1,n,n},{n,n,1,n},{n,n,n,1},{1,1,v,o},{1,v,1,o},{1,o,1,v}}};
  const std::array<std::size_t,10> previous={0,0,n2,0,n3,2*n3,3*n3,0,ov,2*ov};
  for(std::size_t k=0;k<10;++k)
    if(queried[k]!=shapes[k] || bound<8*previous[k]+provider.peak(shapes[k],cuda)) return 3;
  if(bound<16*ov*ov) return 4;
  // With an exact total allowance, only the original component budget remains.
  if(mp2::detail::remaining_force_budget(bound+12345,bound)!=12345) return 5;
  for(auto budget:{bound-1,bound}) {
    bool rejected=false;
    try {(void)mp2::detail::remaining_force_budget(budget,bound);}
    catch(const std::length_error&) {rejected=true;}
    if(!rejected) return 6;
  }
  bool rejected=false;
  try {(void)mp2::detail::streamed_force_workspace_bytes(n,o,[](auto){return 0ULL;});}
  catch(const std::logic_error&) {rejected=true;}
  if(!rejected) return 7;
  rejected=false;
  try {(void)mp2::detail::streamed_force_workspace_bytes(
      std::numeric_limits<std::size_t>::max(),1,[](auto){return 0ULL;});}
  catch(const std::overflow_error&) {rejected=true;}
  if(!rejected) return 8;
  std::vector<double> one(n2),fock(n2),correlation(ov*ov),h(n2);
  trace::live=trace::peak=0;trace::active=true;
  {
    const auto out=mp2::rotation_gradient_streamed(one,fock,correlation,h,provider,n,o,cuda,0);
    if(out.size()!=n2) return 9;
  }
  trace::active=false;
  if(trace::live || trace::peak<32*n3 || trace::peak>bound) {
    std::cerr<<"live="<<trace::live<<" measured="<<trace::peak<<" bound="<<bound<<'\n';
    return 10;
  }
  std::cout<<"measured="<<trace::peak<<" bound="<<bound<<'\n';
}
"""


@pytest.fixture(scope="module")
def workspace_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    header = (ROOT / "src/posthf/mp2_force_workspace.hpp").read_text()
    helper = header[header.index("namespace generativeqc::mp2::detail {") :]
    text = (ROOT / "src/posthf/mp2_gradient.cpp").read_text()
    start = text.index("std::vector<double> rotation_gradient_streamed(")
    rotation = text[start : text.index("\nOrbitalRhs initial_orbital_weights", start)]
    directory = tmp_path_factory.mktemp("mp2-streamed-workspace")
    source = directory / "probe.cpp"
    capacity = (ROOT / "src/posthf/capacity.hpp").read_text()
    begin = capacity.index("inline std::size_t checked_add(")
    arithmetic = capacity[begin : capacity.index("/** CG10", begin)]
    prefix = PREFIX.replace("// CHECKED_ARITHMETIC", arithmetic)
    source.write_text(prefix + helper + ROTATION_PREFIX + rotation + MAIN)
    executable = directory / "probe"
    subprocess.run(
        [compiler, "-std=c++20", "-O0", str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return executable


@pytest.mark.parametrize("n,o", [(8, 1), (12, 5), (24, 12), (32, 1)])
@pytest.mark.parametrize("cuda,staged", [(0, 0), (0, 1), (1, 1)])
def test_streamed_workspace_covers_actual_block_lifetimes(
    workspace_probe: Path, n: int, o: int, cuda: int, staged: int
) -> None:
    result = subprocess.run(
        [str(workspace_probe), str(n), str(o), str(cuda), str(staged)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_force_endpoints_reserve_workspace_before_adjoint_execution() -> None:
    text = (ROOT / "src/posthf/mp2_force.cpp").read_text()
    for name in ("conventional_force_impl", "density_fitted_force_cpu"):
        start = text.index("ConventionalForceResult " + name + "(")
        body = text[start : text.index("\n  return result;", start)]
        assert body.index("streamed_force_workspace_bytes(") < body.index(
            "const auto h = hcore_mo(reference)"
        )
        assert "remaining_force_budget(budget_bytes, streamed_workspace)" in body
        assert "posthf::checked_add(resources.peak_bytes, streamed_workspace)" in body
        assert "endpoint_budget" in body
    native = text[text.index("ConventionalForceResult conventional_force_impl(") :]
    assert "provider.plan(shape, cuda)" in native
    assert "peak - provider_bytes" in native
    assert "endpoint_budget - base_resources.peak_bytes" in native
