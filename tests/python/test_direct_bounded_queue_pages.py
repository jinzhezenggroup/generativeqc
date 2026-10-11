"""Execute production page decoding against an independent pair-inventory oracle."""

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from test_direct_force_schedule_composition import _definition

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_bounded_pages_cover_sparse_sorted_multisystem_inventory(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    """Cover empty/repeated rows, diagonal/tail pages and both consumer widths."""
    queue = (ROOT / "src/scf/cuda/direct_queue_index.cuh").read_text()
    header = (ROOT / "src/scf/cuda/direct_bounded_queue_page.cuh").read_text()
    decoder = header.split("namespace generativeqc::scf::cuda_execution {", 1)[1]
    decoder = decoder.rsplit("}  // namespace", 1)[0]
    definitions = "\n".join(
        _definition(queue, marker)
        for marker in (
            "template <class Offset>",
            "__device__ inline void decode_lower_triangle(",
            "__device__ inline std::int32_t shell_pair_block_quartet_system(",
        )
    )
    source = tmp_path / "pages.cpp"
    source.write_text(
        PREFIX
        + "namespace generativeqc::scf::cuda_execution {\n"
        + definitions
        + decoder
        + "}\n"
        + DRIVER
    )
    executable = native_cxx.build_executable(
        [source],
        tmp_path / "pages",
        compile_args=["-std=c++20", "-O2", f"-I{ROOT / 'src'}"],
    )
    result = subprocess.run(
        [executable], capture_output=True, text=True, check=False, timeout=60
    )
    assert result.returncode == 0, result.stderr


def test_spd_selector_is_captured_by_prepared_force_owner(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    """Compile the real preparation assignment; later environment edits cannot switch it."""
    source = (ROOT / "src/scf/cuda/direct_coulomb.cpp").read_text()
    begin = source.index(
        '  const auto* spd_selector = std::getenv("GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS")'
    )
    end = source.index("  plan->bounded_value_capability", begin)
    assignment = source[begin:end]
    assert source.count('std::getenv("GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS")') == 1
    assert (
        "resolve_direct_spd_force_schedule(p.spd_force_schedule, allow_spatial_default)"
        in source
    )
    probe = tmp_path / "capture.cpp"
    probe.write_text(
        r"""
#include <cassert>
#include <cstdlib>
#include <memory>
#include "scf/cuda/direct_spd_force_schedule.hpp"
using namespace generativeqc::scf::cuda_execution;
struct Plan { DirectSpdForceSchedule spd_force_schedule{}; };
Plan prepare(bool force_capability) {
  auto plan = std::make_unique<Plan>();
"""
        + assignment
        + r"""
  return *plan;
}
int main() {
  static_assert(select_direct_spd_force_schedule("") == DirectSpdForceSchedule::Automatic);
  static_assert(select_direct_spd_force_schedule("0") == DirectSpdForceSchedule::Incumbent);
  static_assert(select_direct_spd_force_schedule("unknown") == DirectSpdForceSchedule::Incumbent);
  static_assert(select_direct_spd_force_schedule("1") == DirectSpdForceSchedule::SpatialReuse);
  static_assert(select_direct_spd_force_schedule("spatial-reuse") == DirectSpdForceSchedule::SpatialReuse);
  for (const bool eligible : {false, true}) {
    assert(resolve_direct_spd_force_schedule(DirectSpdForceSchedule::Incumbent, eligible) == DirectSpdForceSchedule::Incumbent);
    assert(resolve_direct_spd_force_schedule(DirectSpdForceSchedule::SpatialReuse, eligible) == DirectSpdForceSchedule::SpatialReuse);
    assert(resolve_direct_spd_force_schedule(DirectSpdForceSchedule::Automatic, eligible) ==
           (eligible ? DirectSpdForceSchedule::SpatialReuse : DirectSpdForceSchedule::Incumbent));
  }
  assert(unsetenv("GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS") == 0);
  const auto automatic = prepare(true);
  assert(setenv("GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS", "0", 1) == 0);
  const auto baseline = prepare(true);
  const auto value_owner = prepare(false);
  assert(setenv("GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS", "spatial-reuse", 1) == 0);
  const auto spatial = prepare(true);
  assert(unsetenv("GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS") == 0);
  assert(automatic.spd_force_schedule == DirectSpdForceSchedule::Automatic);
  assert(baseline.spd_force_schedule == DirectSpdForceSchedule::Incumbent);
  assert(value_owner.spd_force_schedule == DirectSpdForceSchedule::Incumbent);
  assert(spatial.spd_force_schedule == DirectSpdForceSchedule::SpatialReuse);
}
"""
    )
    executable = native_cxx.build_executable(
        [probe],
        tmp_path / "capture",
        compile_args=["-std=c++20", "-O2", f"-I{ROOT / 'src'}"],
    )
    result = subprocess.run([executable], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_cached_low_ownership_precedes_quartet_screening() -> None:
    """Reject high orders before screening; only order six consumes spatial reuse."""
    source = (ROOT / "src/scf/cuda/direct_force_class_domains.cu").read_text()
    kernel = _definition(
        source,
        "__launch_bounds__(kBoundedDirectForceThreads, 1) void direct_force_spatial_spd_low_kernel(",
    )
    stages = (
        "if (active",
        "bounded_direct_block_pair_survives_screening",
        "bounded_direct_queue_candidate_page",
        "for (auto packet",
        "direct_shell_class_angular_order(shell_class) <= 6U",
        "direct_shell_quartet_survives_screening",
        "profile_bounded_direct_shell_quartet",
        "contract_bounded_spd_low_queue<Unrestricted, Mode, true>",
    )
    positions = [kernel.index(stage) for stage in stages]
    assert positions == sorted(positions)
    drain = _definition(
        source,
        "__device__ __forceinline__ void contract_bounded_spd_low_queue(",
    )
    assert "direct_force_task_shell_class(batch, task)) != 6U" in drain
    assert (
        "Unrestricted, 6U, Mode == DirectForceOutputMode::Separate, SpatialReuse"
        in drain
    )


PREFIX = r"""
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <numeric>
#include <tuple>
#include <vector>
#define __host__
#define __device__
#include "scf/direct_block_domain.hpp"
#include "scf/direct_task_layout.hpp"
using std::min;
using std::sqrt;
struct DeviceBatch {
  std::int32_t batch_size{};
  const std::uint64_t* system_shell_pair_offsets{};
  const std::uint64_t* system_shell_pair_block_offsets{};
  const std::uint64_t* system_shell_pair_block_quartet_offsets{};
  std::size_t total_shell_pair_block_quartets{};
};
"""

DRIVER = r"""
using namespace generativeqc::scf::cuda_execution;
namespace detail = generativeqc::scf::detail;
using Key = std::tuple<unsigned, unsigned, unsigned>;
using Inventory = std::map<Key, unsigned>;
void exercise(const std::vector<unsigned>& counts, unsigned seed, bool indexed, unsigned width) {
  const auto systems = counts.size();
  const auto block_size = detail::kBoundedDirectShellPairBlockSize;
  std::vector<std::uint64_t> pairs(systems + 1), blocks(systems + 1), products(systems + 1);
  std::vector<unsigned> order;
  for (unsigned system = 0; system < systems; ++system) {
    pairs[system + 1] = pairs[system] + counts[system];
    const auto block_count = (counts[system] + block_size - 1) / block_size;
    blocks[system + 1] = blocks[system] + block_count;
    products[system + 1] = products[system] + block_count * (block_count + 1) / 2;
    for (unsigned position = 0; position < counts[system]; ++position)
      order.push_back(pairs[system] + position);
    auto first = order.begin() + pairs[system];
    if (counts[system]) {
      std::reverse(first, order.end());
      std::rotate(first, first + seed % counts[system], order.end());
    }
  }
  std::vector<std::uint64_t> prefix(blocks.back() + 1);
  for (unsigned system = 0; system < systems; ++system)
    for (auto row = blocks[system]; row < blocks[system + 1]; ++row) {
      const auto local_row = row - blocks[system];
      const auto kept = indexed ? (local_row + seed + system) % (local_row + 2) : local_row + 1;
      prefix[row + 1] = prefix[row] + kept;
    }
  const DeviceBatch batch{static_cast<std::int32_t>(systems), pairs.data(), blocks.data(),
                          products.data(), static_cast<std::size_t>(products.back())};
  const detail::BoundedDirectBlockDomain domain{
      indexed ? prefix.data() : nullptr, blocks.back(), prefix.back()};
  const auto claims = indexed ? prefix.back() * detail::kBoundedDirectIndexedCandidatePages
                              : products.back();
  Inventory actual, expected;
  std::size_t visited_candidates = 0, empty_pages = 0;
  for (unsigned worker = 0; worker < 7; ++worker) {
    BoundedDirectQueuePage page{};
    for (std::uint64_t claim = worker; bounded_direct_queue_page(batch, domain, claim, page);
         claim += 7) {
      assert(claim < claims);
      assert(page.system >= 0 && static_cast<unsigned>(page.system) < systems);
      assert(page.first_block >= blocks[page.system] && page.first_block < blocks[page.system + 1]);
      assert(page.second_block >= blocks[page.system] && page.second_block <= page.first_block);
      assert(page.same_block == (page.first_block == page.second_block));
      if (page.candidate_begin >= page.candidate_end) ++empty_pages;
      if (indexed) assert(page.candidate_end <= page.candidate_begin + 64);
      for (auto packet = page.candidate_begin; packet < page.candidate_end; packet += width)
        for (unsigned lane = 0; lane < width && packet + lane < page.candidate_end; ++lane) {
          const auto candidate = packet + lane;
          std::size_t first_local{}, second_local{};
          if (page.same_block) decode_lower_triangle(candidate, first_local, second_local);
          else {
            first_local = candidate / page.second_count;
            second_local = candidate % page.second_count;
          }
          const auto first = order.at(page.first_ordered_begin + first_local);
          const auto second = order.at(page.second_ordered_begin + second_local);
          ++actual[{page.system, std::max(first, second), std::min(first, second)}];
          ++visited_candidates;
        }
    }
  }
  for (unsigned system = 0; system < systems; ++system)
    for (unsigned first = 0; first < counts[system]; ++first)
      for (unsigned second = 0; second <= first; ++second) {
        const auto row = blocks[system] + first / block_size;
        if (indexed && second / block_size >= prefix[row + 1] - prefix[row]) continue;
        const auto physical_first = order[pairs[system] + first];
        const auto physical_second = order[pairs[system] + second];
        ++expected[{system, std::max(physical_first, physical_second),
                            std::min(physical_first, physical_second)}];
      }
  assert(actual == expected && visited_candidates == expected.size());
  for (const auto& [key, multiplicity] : actual) assert(multiplicity == 1);
  BoundedDirectQueuePage exhausted{};
  assert(!bounded_direct_queue_page(batch, domain, claims, exhausted));
  assert(!bounded_direct_queue_page(batch, domain, claims + 1, exhausted));
  if (indexed && seed == 1 && !actual.empty()) assert(empty_pages > 0);
}
int main() {
  for (const auto& counts : {std::vector<unsigned>{0,0,0},
                            std::vector<unsigned>{0,1,31,32,33,65,0,129,0}})
    for (unsigned seed = 0; seed < 7; ++seed)
      for (bool indexed : {false,true})
        for (unsigned width : {128U,256U}) exercise(counts, seed, indexed, width);
}
"""
