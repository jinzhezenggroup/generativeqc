"""Compile the actual class-major page decoder against an independent inventory."""

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_physical_pair_ids_keep_unsigned_provider_independent_orientation(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    """Compile actual selection with signed-only provider overloads and large IDs."""
    source = (ROOT / "src/scf/cuda/direct_order_seven_force.cu").read_text()
    begin = source.index("const auto bra = topology.pair_order[page.bra];")
    end = source.index("if (direct_shell_quartet_survives_screening", begin)
    selection = source[begin:end]
    begin = source.index(
        "const ActiveShellQuartetTile task{first_pair, second_pair, 0U};"
    )
    task = source[begin : source.index(";", begin) + 1]
    probe = tmp_path / "unsigned-pairs.cpp"
    probe.write_text(
        r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <type_traits>
int max(int first,int second) { return first>second ? first : second; }
int min(int first,int second) { return first<second ? first : second; }
struct ActiveShellQuartetTile { std::uint32_t first_pair,second_pair,tile; };
int main() {
  const std::array<std::uint32_t,5> ids{0U,1U,0x7fffffffU,0x80000000U,0xffffffffU};
  for (auto first : ids) for (auto second : ids) {
    const std::uint32_t pairs[]{first,second};
    struct { const std::uint32_t* pair_order; } topology{pairs};
    struct { std::uint32_t bra=0,ket_begin=1; } page;
    struct { unsigned x=0; } threadIdx;
"""
        + selection
        + task
        + r"""
    static_assert(std::is_same_v<decltype(first_pair),const std::uint32_t>);
    static_assert(std::is_same_v<decltype(second_pair),const std::uint32_t>);
    assert(task.first_pair==std::max(first,second));
    assert(task.second_pair==std::min(first,second) && task.tile==0U);
  }
}
"""
    )
    executable = native_cxx.build_executable(
        [probe],
        tmp_path / "unsigned-pairs",
        compile_args=["-std=c++17", "-Werror=narrowing"],
    )
    subprocess.run([str(executable)], check=True, timeout=10)


def test_order_seven_pages_own_exact_same_system_rectangle(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    """Check empty systems, tails, reordered physical IDs and 64-bit claims."""
    probe = tmp_path / "pages.cpp"
    probe.write_text(DRIVER)
    executable = native_cxx.build_executable(
        [probe],
        tmp_path / "pages",
        compile_args=["-std=c++20", "-O2", f"-I{ROOT / 'src'}"],
    )
    result = subprocess.run([executable], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


DRIVER = r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <random>
#include <set>
#include <utility>
#include <vector>
#define __host__
#define __device__
#include "scf/cuda/direct_order_seven_pages.cuh"
using namespace generativeqc::scf::cuda_execution;
using Pair = std::pair<std::uint32_t, std::uint32_t>;

void check(const std::vector<std::uint32_t>& bra_counts,
           const std::vector<std::uint32_t>& ket_counts, std::mt19937& random) {
  const auto systems = static_cast<std::uint32_t>(bra_counts.size());
  const auto stride = systems + 1U;
  std::vector<std::uint32_t> offsets(10U * stride);
  std::uint32_t total = 0;
  for (unsigned pair_class = 0; pair_class < 10; ++pair_class) {
    offsets[pair_class * stride] = total;
    for (unsigned system = 0; system < systems; ++system) {
      total += pair_class == 5 ? bra_counts[system] : pair_class == 4 ? ket_counts[system] : 1;
      offsets[pair_class * stride + system + 1U] = total;
    }
  }
  std::vector<std::uint32_t> pairs(total);
  for (std::uint32_t index = 0; index < total; ++index) pairs[index] = index;
  std::shuffle(pairs.begin(), pairs.end(), random);
  std::set<Pair> expected;
  std::uint64_t expected_pages = 0;
  for (unsigned system = 0; system < systems; ++system) {
    expected_pages += std::uint64_t{bra_counts[system]} * ((ket_counts[system] + 31U) / 32U);
    for (auto bra = offsets[5U * stride + system]; bra < offsets[5U * stride + system + 1U]; ++bra)
      for (auto ket = offsets[4U * stride + system]; ket < offsets[4U * stride + system + 1U]; ++ket)
        expected.emplace(std::max(pairs[bra], pairs[ket]), std::min(pairs[bra], pairs[ket]));
  }
  for (unsigned workers : {1U, 7U, 32U}) {
    std::set<Pair> actual;
    std::uint64_t pages = 0;
    for (unsigned worker = 0; worker < workers; ++worker) {
      DirectOrderSevenPagePosition position{};
      DirectOrderSevenPairPage page{};
      for (std::uint64_t claim = worker;
           direct_order_seven_pair_page(offsets.data(), systems, claim, position, page);
           claim += workers) {
        ++pages;
        assert(page.system < systems);
        assert(page.bra >= offsets[5U * stride + page.system]);
        assert(page.bra < offsets[5U * stride + page.system + 1U]);
        assert(page.ket_begin >= offsets[4U * stride + page.system]);
        assert(page.ket_end <= offsets[4U * stride + page.system + 1U]);
        assert(page.ket_begin < page.ket_end && page.ket_end - page.ket_begin <= 32U);
        for (auto ket = page.ket_begin; ket < page.ket_end; ++ket)
          assert(actual.emplace(std::max(pairs[page.bra], pairs[ket]),
                                std::min(pairs[page.bra], pairs[ket])).second);
      }
      assert(!direct_order_seven_pair_page(offsets.data(), systems, expected_pages + 32U,
                                          position, page));
    }
    assert(pages == expected_pages && actual == expected);
  }
}

int main() {
  std::mt19937 random(4917);
  check({}, {}, random);
  check({0, 3, 0, 2, 0}, {7, 0, 0, 65, 0}, random);
  for (unsigned ket : {0U, 1U, 31U, 32U, 33U, 63U, 64U, 65U})
    check({2}, {ket}, random);
  for (unsigned trial = 0; trial < 100; ++trial) {
    const auto systems = 1U + random() % 8U;
    std::vector<std::uint32_t> bra(systems), ket(systems);
    for (unsigned system = 0; system < systems; ++system) {
      bra[system] = random() % 11U;
      ket[system] = random() % 100U;
    }
    check(bra, ket, random);
  }
  const auto maximum = std::numeric_limits<std::uint32_t>::max();
  for (auto counts : {Pair{1000000U,1000000U}, Pair{1U,maximum-1U}}) {
    std::vector<std::uint32_t> offsets(20);
    offsets[8] = 0; offsets[9] = counts.second;
    offsets[10] = counts.second; offsets[11] = counts.first + counts.second;
    const std::uint64_t ket_pages = (std::uint64_t{counts.second} + 31U) / 32U;
    const std::uint64_t pages = std::uint64_t{counts.first} * ket_pages;
    for (auto claim : {std::uint64_t{0}, pages / 2, pages - 1}) {
      DirectOrderSevenPagePosition position{};
      DirectOrderSevenPairPage page{};
      assert(direct_order_seven_pair_page(offsets.data(), 1U, claim, position, page));
      assert(page.bra == counts.second + claim / ket_pages);
      assert(page.ket_begin == (claim % ket_pages) * 32U);
      assert(page.ket_end == std::min(std::uint64_t{counts.second},
                                      std::uint64_t{page.ket_begin} + 32U));
      assert(!direct_order_seven_pair_page(offsets.data(), 1U, pages, position, page));
    }
  }
}
"""
