"""Execute production mixed-basis ownership and compact class-page addressing."""

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from test_direct_force_schedule_composition import _definition

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_class_domains_cover_each_physical_quartet_once(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    """Cover empty/ragged systems, page boundaries, f exclusion and large IDs."""
    bounded = (ROOT / "src/scf/cuda/direct_bounded_fallback.cu").read_text()
    owner = _definition(bounded, "bool bounded_direct_angular_owner(")
    source = tmp_path / "pages.cpp"
    source.write_text(PREFIX + "template<int AngularOrder>\n" + owner + DRIVER)
    executable = native_cxx.build_executable(
        [source],
        tmp_path / "pages",
        compile_args=["-std=c++20", "-O2", f"-I{ROOT / 'src'}"],
    )
    result = subprocess.run([executable], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_class_page_decoder_remains_a_schedule_only_leaf(tmp_path: Path) -> None:
    """Address ownership must not acquire an integral or runtime allocator dependency."""
    from tools.check_scf_structure import audit_scf_structure

    source = tmp_path / "src/scf/cuda/direct_force_class_pages.cuh"
    source.parent.mkdir(parents=True)
    (source.parent / "direct_force_quartet.cuh").write_text("#pragma once\n")
    source.write_text('#include "scf/cuda/direct_force_quartet.cuh"\n')
    errors = audit_scf_structure(tmp_path)["errors"]
    assert len(errors) == 1 and "forbidden" in errors[0]


PREFIX = r"""
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <map>
#include <tuple>
#include <vector>
#define __host__
#define __device__
#include "scf/cuda/direct_force_class_pages.cuh"
using namespace generativeqc::scf::cuda_execution;
struct DeviceBatch {
  const unsigned* shell_angular;
  const unsigned* shell_pair_first;
  const unsigned* shell_pair_second;
};
"""

DRIVER = r"""
using Key = std::tuple<unsigned,unsigned,unsigned>;
template<DirectForceClassDomain Domain>
void enumerate(const std::vector<std::uint32_t>& offsets, unsigned systems,
               std::map<Key,unsigned>& seen) {
  for (unsigned worker = 0; worker < 7; ++worker) {
    DirectForceClassPagePosition position{};
    DirectForceClassPage page{};
    for (std::uint64_t claim = worker;
         direct_force_class_pair_page<Domain>(offsets.data(), systems, claim, position, page);
         claim += 7U) {
      assert(page.ket_end > page.ket_begin);
      assert(page.ket_end - page.ket_begin <= direct_force_class_page_size(Domain));
      for (unsigned ket = page.ket_begin; ket < page.ket_end; ++ket)
        ++seen[{page.system,page.bra,ket}];
    }
  }
}
int main() {
  constexpr unsigned systems = 4, stride = systems + 1;
  for (unsigned seed = 0; seed < 9; ++seed) {
    std::vector<std::uint32_t> offsets(10 * stride);
    unsigned next = 0;
    for (unsigned pair_class = 0; pair_class < 10; ++pair_class) {
      offsets[pair_class * stride] = next;
      for (unsigned system = 0; system < systems; ++system) {
        const unsigned counts[]{0U,1U,31U,32U,33U,65U,127U,128U,129U};
        next += system == 0 ? 0U : counts[(pair_class + system + seed) % 9U];
        offsets[pair_class * stride + system + 1] = next;
      }
    }
    std::map<Key,unsigned> seen, expected;
    enumerate<DirectForceClassDomain::LowOrder>(offsets,systems,seen);
    enumerate<DirectForceClassDomain::WeightedFour>(offsets,systems,seen);
    enumerate<DirectForceClassDomain::WeightedFive>(offsets,systems,seen);
    enumerate<DirectForceClassDomain::Cooperative>(offsets,systems,seen);
    enumerate<DirectForceClassDomain::Materialized>(offsets,systems,seen);
    for (unsigned system = 0; system < systems; ++system)
      for (unsigned first = 0; first < 6; ++first)
        for (unsigned second = 0; second <= first; ++second)
          for (unsigned bra = offsets[first*stride+system];
               bra < offsets[first*stride+system+1]; ++bra)
            for (unsigned ket = offsets[second*stride+system];
                 ket < offsets[second*stride+system+1] && (first != second || ket <= bra); ++ket)
              ++expected[{system,bra,ket}];
    assert(seen == expected);
  }
  const unsigned angular[]{0U,1U,2U,3U};
  const unsigned first_shell[]{0U,1U,1U,2U,2U,2U,3U,3U,3U,3U};
  const unsigned second_shell[]{0U,0U,1U,0U,1U,2U,0U,1U,2U,3U};
  const DeviceBatch batch{angular,first_shell,second_shell};
  std::array<unsigned,5> class_counts{};
  for (unsigned first = 0; first < 10; ++first)
    for (unsigned second = 0; second <= first; ++second) {
      const auto domain = direct_force_class_domain(first,second);
      const bool fallback = bounded_direct_angular_owner<-4>(batch,first,second);
      assert(fallback == (domain == DirectForceClassDomain::Fallback));
      if (!fallback) ++class_counts[static_cast<unsigned>(domain)];
    }
  assert((class_counts == std::array<unsigned,5>{8U,5U,3U,4U,1U}));
  const auto count = std::numeric_limits<std::uint32_t>::max();
  std::array<std::uint32_t,12> large{};
  large[11] = count;
  const auto pages = direct_force_triangle_page_prefix(count);
  DirectForceClassPagePosition position{};
  DirectForceClassPage page{};
  assert(direct_force_class_pair_page<DirectForceClassDomain::Materialized>(
      large.data(),1,pages-1U,position,page));
  assert(page.bra == count-1U && page.ket_end == count);
  assert(!direct_force_class_pair_page<DirectForceClassDomain::Materialized>(
      large.data(),1,pages,position,page));
}
"""
