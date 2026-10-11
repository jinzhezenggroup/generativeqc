"""Execute cached-low routing and real Cartesian tile indexing without a GPU."""

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from test_direct_bounded_queue_pages import PREFIX as PAGE_PREFIX
from test_direct_force_schedule_composition import _definition

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_low_drain_preserves_scalar_and_multitile_warp_inventory(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    """Check both spins/layouts, diagonal packets and the >256-component tails."""
    source = (ROOT / "src/scf/cuda/direct_force_class_domains.cu").read_text()
    quartet = (ROOT / "src/scf/cuda/direct_force_quartet.cuh").read_text()
    drain = _definition(
        source,
        "template <bool Unrestricted, DirectForceOutputMode Mode, bool SpatialReuse = false>\n"
        "__device__ __forceinline__ void contract_bounded_spd_low_queue(",
    )
    view_begin = quartet.index("  constexpr std::size_t subtiles_per_tile =")
    view_end = quartet.index("  const std::size_t first_pair =", view_begin)
    ordinal_begin = quartet.index("  const std::size_t ordinal =", view_end)
    ordinal_end = quartet.index("  if (ordinal < ao_quartet_count)", ordinal_begin)
    consumer = (
        CONSUMER_PREFIX
        + quartet[view_begin:view_end]
        + "\n  const auto ao_quartet_count = components(batch, task);\n"
        + quartet[ordinal_begin:ordinal_end]
        + CONSUMER_SUFFIX
    )
    probe = tmp_path / "drain.cpp"
    probe.write_text(DRAIN_PREFIX + consumer + drain + DRAIN_DRIVER)
    executable = native_cxx.build_executable(
        [probe],
        tmp_path / "drain",
        compile_args=["-std=c++20", "-O2", f"-I{ROOT / 'src'}"],
    )
    result = subprocess.run(
        [executable], capture_output=True, text=True, check=False, timeout=60
    )
    assert result.returncode == 0, result.stderr


def test_block_decode_does_not_need_rejected_pair_geometry(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    """A null pair-range view is safe until the actual kernel passes coarse gates."""
    queue = (ROOT / "src/scf/cuda/direct_queue_index.cuh").read_text()
    header = (ROOT / "src/scf/cuda/direct_bounded_queue_page.cuh").read_text()
    source = (ROOT / "src/scf/cuda/direct_force_class_domains.cu").read_text()
    kernel = _definition(source, "void direct_force_spatial_spd_low_kernel(")
    assert kernel.index("bounded_direct_queue_block_page") < kernel.index("if (active")
    assert kernel.index("if (active") < kernel.index(
        "bounded_direct_block_pair_survives_screening"
    )
    assert kernel.index("bounded_direct_block_pair_survives_screening") < kernel.index(
        "bounded_direct_queue_candidate_page"
    )
    assert kernel.index("direct_shell_class_angular_order(shell_class) <= 6U") < (
        kernel.index("direct_shell_quartet_survives_screening")
    )
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
    probe = tmp_path / "staged.cpp"
    probe.write_text(
        PAGE_PREFIX
        + "namespace generativeqc::scf::cuda_execution {\n"
        + definitions
        + decoder
        + "}\n"
        + STAGED_DRIVER
    )
    executable = native_cxx.build_executable(
        [probe],
        tmp_path / "staged",
        compile_args=["-std=c++20", "-O2", f"-I{ROOT / 'src'}"],
    )
    result = subprocess.run(
        [executable], capture_output=True, text=True, check=False, timeout=60
    )
    assert result.returncode == 0, result.stderr


DRAIN_PREFIX = r"""
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <map>
#include <tuple>
#include <vector>
#define __device__
#define __forceinline__ inline
#include "scf/direct_task_layout.hpp"
namespace detail = generativeqc::scf::detail;
enum class DirectForceOutputMode { Combined, Separate };
struct ActiveShellQuartetTile {
  unsigned first_pair, second_pair, tile;
  bool operator==(const ActiveShellQuartetTile&) const = default;
};
struct DeviceBatch { std::vector<unsigned> pair_orders, pair_counts; };
struct Index { unsigned x; } threadIdx{}, blockDim{128};
void __syncthreads() {}
unsigned direct_force_task_shell_class(const DeviceBatch& batch, ActiveShellQuartetTile task) {
  return batch.pair_orders.at(task.first_pair) + batch.pair_orders.at(task.second_pair);
}
unsigned direct_shell_class_angular_order(unsigned shell_class) { return shell_class; }
std::size_t shell_ao_pair_count(const DeviceBatch& batch, unsigned pair) {
  return batch.pair_counts.at(pair);
}
std::size_t components(const DeviceBatch& batch, ActiveShellQuartetTile task) {
  const auto first = batch.pair_counts.at(task.first_pair);
  const auto second = batch.pair_counts.at(task.second_pair);
  return task.first_pair == task.second_pair ? first * (first + 1U) / 2U : first * second;
}
using ScalarKey = std::tuple<unsigned, unsigned, unsigned>;
using AoKey = std::tuple<unsigned, unsigned, std::size_t, std::size_t>;
std::map<ScalarKey, unsigned> scalar_visits;
std::map<AoKey, unsigned> ao_visits;
bool expected_spin{}, expected_separate{}, expected_reuse{};
template <bool Unrestricted, DirectForceOutputMode Mode, class... Arguments>
void contract_direct_force_precontracted_task(
    const DeviceBatch& batch, ActiveShellQuartetTile task, Arguments...) {
  assert(Unrestricted == expected_spin && (Mode == DirectForceOutputMode::Separate) == expected_separate);
  assert(direct_force_task_shell_class(batch, task) <= 3U);
  ++scalar_visits[{0U, task.first_pair, task.second_pair}];
}
template <bool Unrestricted, DirectForceOutputMode Mode, class... Arguments>
void contract_two_electron_force_order4_sources(
    unsigned shell_class, const DeviceBatch&, ActiveShellQuartetTile task, Arguments...) {
  assert(Unrestricted == expected_spin && (Mode == DirectForceOutputMode::Separate) == expected_separate);
  assert(shell_class == 4U);
  ++scalar_visits[{4U, task.first_pair, task.second_pair}];
}
template <bool Unrestricted, DirectForceOutputMode Mode, class... Arguments>
void contract_two_electron_force_order5_sources(
    unsigned shell_class, const DeviceBatch&, ActiveShellQuartetTile task, Arguments...) {
  assert(Unrestricted == expected_spin && (Mode == DirectForceOutputMode::Separate) == expected_separate);
  assert(shell_class == 5U);
  ++scalar_visits[{5U, task.first_pair, task.second_pair}];
}
"""

CONSUMER_PREFIX = r"""
template <bool Unrestricted, unsigned AngularOrder, bool SeparateSources, bool SpatialReuse>
void contract_two_electron_force_quartet_subtile_scaled(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, double,
    const double*, const double*, const std::uint8_t*, double*, std::uint64_t,
    double, double, std::size_t active_subtile, unsigned ao_quartet_lane) {
  static_assert(AngularOrder == 6U);
  assert(Unrestricted == expected_spin && SeparateSources == expected_separate);
  assert(SpatialReuse == expected_reuse);
  assert(*active_shell_quartet_tile_count == 1U);
"""

CONSUMER_SUFFIX = r"""
  if (ordinal >= ao_quartet_count) return;
  std::size_t first_ao{}, second_ao{};
  if (task.first_pair == task.second_pair) {
    while ((first_ao + 1U) * (first_ao + 2U) / 2U <= ordinal) ++first_ao;
    second_ao = ordinal - first_ao * (first_ao + 1U) / 2U;
  } else {
    first_ao = ordinal / batch.pair_counts.at(task.second_pair);
    second_ao = ordinal % batch.pair_counts.at(task.second_pair);
  }
  ++ao_visits[{task.first_pair, task.second_pair, first_ao, second_ao}];
}
"""

DRAIN_DRIVER = r"""
template <bool Unrestricted, DirectForceOutputMode Mode, bool SpatialReuse>
void exercise() {
  expected_spin = Unrestricted;
  expected_separate = Mode == DirectForceOutputMode::Separate;
  expected_reuse = SpatialReuse;
  const DeviceBatch batch{{0, 1, 2, 2, 3, 4, 3}, {1, 3, 9, 6, 18, 36, 18}};
  std::vector<ActiveShellQuartetTile> queue;
  for (unsigned repeat = 0; repeat < 8; ++repeat)
    for (unsigned first = 0; first < batch.pair_orders.size(); ++first)
      for (unsigned second = 0; second <= first; ++second)
        queue.push_back({first, second, 0U});
  const auto original = queue;
  for (unsigned count : {0U, 1U, 7U, 28U, 129U, 224U}) {
    scalar_visits.clear();
    ao_visits.clear();
    std::map<ScalarKey, unsigned> expected_scalar;
    std::map<AoKey, unsigned> expected_ao;
    for (unsigned slot = 0; slot < count; ++slot) {
      const auto task = queue.at(slot);
      const auto order = batch.pair_orders[task.first_pair] + batch.pair_orders[task.second_pair];
      if (order < 6U) {
        ++expected_scalar[{order <= 3U ? 0U : order, task.first_pair, task.second_pair}];
      } else if (order == 6U) {
        for (unsigned first = 0; first < batch.pair_counts[task.first_pair]; ++first)
          for (unsigned second = 0; second < batch.pair_counts[task.second_pair]; ++second) {
            if (task.first_pair == task.second_pair && second > first) continue;
            ++expected_ao[{task.first_pair, task.second_pair, first, second}];
          }
      }
    }
    for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x)
      contract_bounded_spd_low_queue<Unrestricted, Mode, SpatialReuse>(
          batch, queue.data(), count, 0.0, nullptr, nullptr, nullptr, nullptr, 1.7, -0.23);
    assert(scalar_visits == expected_scalar);
    assert(ao_visits == expected_ao);
    assert(queue == original);
  }
  assert(components(batch, {5, 2, 0}) == 324U);
  assert(components(batch, {6, 4, 0}) == 324U);
  assert(components(batch, {4, 4, 0}) == 171U);
}
int main() {
  exercise<false, DirectForceOutputMode::Combined, false>();
  exercise<true, DirectForceOutputMode::Combined, false>();
  exercise<false, DirectForceOutputMode::Separate, false>();
  exercise<true, DirectForceOutputMode::Separate, false>();
  exercise<false, DirectForceOutputMode::Combined, true>();
  exercise<true, DirectForceOutputMode::Combined, true>();
  exercise<false, DirectForceOutputMode::Separate, true>();
  exercise<true, DirectForceOutputMode::Separate, true>();
}
"""

STAGED_DRIVER = r"""
using namespace generativeqc::scf::cuda_execution;
namespace detail = generativeqc::scf::detail;
int main() {
  const std::uint64_t blocks[]{0, 1, 2}, products[]{0, 1, 2}, pairs[]{0, 3, 4};
  const std::uint64_t prefix[]{0, 1, 2};
  for (const bool indexed : {false, true}) {
    const detail::BoundedDirectBlockDomain domain{indexed ? prefix : nullptr, 2U, 2U};
    const DeviceBatch batch{2, pairs, blocks, products, 2U};
    auto block_only = batch;
    block_only.system_shell_pair_offsets = nullptr;
    const auto claims = 2U * (indexed ? detail::kBoundedDirectIndexedCandidatePages : 1U);
    for (std::uint64_t ordinal = 0; ordinal < claims; ++ordinal) {
      BoundedDirectQueuePage staged{}, complete{};
      staged.candidate_end = 987U;
      staged.first_ordered_begin = 654U;
      assert(bounded_direct_queue_block_page(block_only, domain, ordinal, staged));
      assert(staged.candidate_end == 987U && staged.first_ordered_begin == 654U);
      assert(bounded_direct_queue_page(batch, domain, ordinal, complete));
      bounded_direct_queue_candidate_page(batch, indexed, staged);
      assert(std::tie(staged.system, staged.first_block, staged.second_block,
                      staged.first_ordered_begin, staged.second_ordered_begin, staged.second_count,
                      staged.candidate_begin, staged.candidate_end, staged.same_block) ==
             std::tie(complete.system, complete.first_block, complete.second_block,
                      complete.first_ordered_begin, complete.second_ordered_begin, complete.second_count,
                      complete.candidate_begin, complete.candidate_end, complete.same_block));
    }
    BoundedDirectQueuePage exhausted{};
    assert(!bounded_direct_queue_block_page(block_only, domain, claims, exhausted));
  }
}
"""
