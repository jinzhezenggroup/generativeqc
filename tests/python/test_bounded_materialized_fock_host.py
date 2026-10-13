"""Execute bounded Fock traversal with host substitutes for device consumers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_bounded_materialized_class_traversal(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Check the actual kernel's domains, screening and unsigned pair identity.

    Device contraction and barriers are substituted; this is a traversal and
    compilation contract, not a substitute for real-GPU numerical/race tests.
    Signed-only min/max model the CuMetal overload set that exposed narrowing.
    """
    source = (ROOT / "src/scf/cuda/direct_angular_fock.cu").read_text()
    start = source.index("template <bool Unrestricted, unsigned AngularOrder>")
    end = source.index("cudaError_t launch_bounded_materialized_fock(", start)
    probe = tmp_path / "bounded_traversal.cpp"
    probe.write_text(PREFIX + source[start:end] + SUFFIX)
    binary = native_cxx.build_executable(
        [probe],
        tmp_path / "bounded_traversal",
        compile_args=["-std=c++20", "-O0", "-Werror=narrowing", "-pthread"],
    )
    subprocess.run([binary], check=True, timeout=30)


PREFIX = r"""
#include <cassert>
#include <algorithm>
#include <array>
#include <barrier>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <thread>
#include <utility>
#include <vector>
#define __global__
#define __shared__ static
struct Dim { unsigned x{}; } blockIdx, gridDim;
thread_local Dim threadIdx;
std::barrier block_barrier(32);
std::array<unsigned, 32> votes;
void __syncthreads() { block_barrier.arrive_and_wait(); }
unsigned __ballot_sync(unsigned mask, bool keep) {
  assert(mask == 0xffffffffU);
  votes[threadIdx.x] = keep ? (1U << threadIdx.x) : 0;
  __syncthreads();
  unsigned result = 0;
  for (unsigned vote : votes) result |= vote;
  __syncthreads();
  return result;
}
unsigned __popc(unsigned value) { return std::popcount(value); }
unsigned __ffs(unsigned value) { return std::countr_zero(value) + 1; }
int max(int first, int second) { return first > second ? first : second; }
int min(int first, int second) { return first < second ? first : second; }
void atomicAdd(unsigned long long* value, unsigned long long increment) { *value += increment; }
struct DeviceBatch {};
struct ShellPairDensityBounds {};
struct GeneratedShellPairStream {
  std::int32_t batch_size;
  const std::uint32_t* pair_class_offsets;
  const std::uint32_t* pair_order;
  const double* shell_pair_bounds{};
  const ShellPairDensityBounds* shell_pair_density_bounds{};
};
struct ActiveShellQuartetTile { std::uint32_t first_pair, second_pair, tile; };
template <unsigned> struct MaterializedDirectPairRecurrence {};
constexpr unsigned canonical_materialized_component_lanes(unsigned) { return 32; }
constexpr unsigned canonical_materialized_component_slots(unsigned) { return 1; }
enum class DirectScreeningPurpose { Fock };
bool survive = true;
bool alternate_screening = false;
template <bool, DirectScreeningPurpose>
bool direct_shell_quartet_survives_screening(DeviceBatch, std::uint32_t first, std::uint32_t second,
                                           double, const double*, const ShellPairDensityBounds*,
                                           std::nullptr_t) {
  return survive && (!alternate_screening || ((first ^ second) & 1U));
}
void decode_lower_triangle(std::size_t ordinal, std::size_t& first, std::size_t& second) {
  first = static_cast<std::size_t>((std::sqrt(8.0L * ordinal + 1) - 1) / 2);
  while (first * (first + 1) / 2 > ordinal) --first;
  while ((first + 1) * (first + 2) / 2 <= ordinal) ++first;
  second = ordinal - first * (first + 1) / 2;
}
std::vector<std::pair<std::uint32_t, std::uint32_t>> visited;
template <bool, unsigned Order, unsigned Slots, unsigned Lanes, typename Scalar, bool Cooperative,
          bool Convolved>
void contract_materialized_direct_pair_fock(DeviceBatch, ActiveShellQuartetTile task, double,
    const double*, const double*, const std::uint8_t*, double*, std::nullptr_t,
    MaterializedDirectPairRecurrence<Order>&) {
  assert(task.tile == 0);
  assert(task.first_pair >= task.second_pair);
  if (threadIdx.x == 0) visited.emplace_back(task.first_pair, task.second_pair);
}
"""

SUFFIX = r"""
template <bool Unrestricted>
void run(GeneratedShellPairStream topology, unsigned bra, unsigned ket,
         const std::uint8_t* active = nullptr, unsigned block = 0, unsigned stride = 1) {
  visited.clear();
  blockIdx.x = block; gridDim.x = stride;
  unsigned long long work = 0;
  std::vector<std::thread> threads;
  for (unsigned lane = 0; lane < 32; ++lane)
    threads.emplace_back([&, lane] {
      threadIdx.x = lane;
      bounded_materialized_fock_kernel<Unrestricted, 3>(
          {}, &topology, bra, ket, 0, nullptr, nullptr, active, nullptr, &work);
    });
  for (auto& thread : threads) thread.join();
  assert(work == visited.size());
}
template <bool Unrestricted> void check() {
  const std::uint32_t maximum = std::numeric_limits<std::uint32_t>::max();
  // Two systems, two classes, deliberately unsorted full-width pair identities.
  const std::uint32_t pairs[]{maximum, 0, maximum - 1, 4, 2, 3};
  const std::uint32_t offsets[]{0, 2, 3, 3, 5, 6};
  const GeneratedShellPairStream topology{2, offsets, pairs};
  const std::uint8_t active[]{1, 0};
  run<Unrestricted>(topology, 0, 0, active);
  assert((visited == std::vector<std::pair<std::uint32_t, std::uint32_t>>{
      {maximum, maximum}, {maximum, 0}, {0, 0}}));
  run<Unrestricted>(topology, 0, 1, active);
  assert((visited == std::vector<std::pair<std::uint32_t, std::uint32_t>>{
      {maximum, 4}, {maximum, 2}, {4, 0}, {2, 0}}));
  run<Unrestricted>(topology, 0, 1);
  assert(visited.size() == 5 && visited.back().first == maximum - 1);
  alternate_screening = true;
  run<Unrestricted>(topology, 0, 1, active);
  assert((visited == std::vector<std::pair<std::uint32_t, std::uint32_t>>{
      {maximum, 4}, {maximum, 2}}));
  alternate_screening = false;
  run<Unrestricted>(topology, 0, 1, active, 1, 3);
  assert(visited.size() == 1 && visited[0].second == 2);
  survive = false;
  run<Unrestricted>(topology, 0, 1);
  assert(visited.empty());
  survive = true;
  const std::uint32_t empty_offsets[]{0, 0, 0, 0, 0, 0};
  run<Unrestricted>({2, empty_offsets, pairs}, 0, 1);
  assert(visited.empty());
  std::vector<std::uint32_t> paged_pairs(16);
  for (std::size_t index = 0; index < paged_pairs.size(); ++index)
    paged_pairs[index] = static_cast<std::uint32_t>(index);
  const std::uint32_t paged_offsets[]{0, 8, 8, 16};
  const GeneratedShellPairStream paged_topology{1, paged_offsets, paged_pairs.data()};
  run<Unrestricted>(paged_topology, 0, 1);
  const auto complete = visited;
  assert(complete.size() == 64);
  for (unsigned workers : {2U, 3U, 5U, 128U}) {
    std::vector<std::pair<std::uint32_t, std::uint32_t>> combined;
    for (unsigned worker = 0; worker < workers; ++worker) {
      run<Unrestricted>(paged_topology, 0, 1, nullptr, worker, workers);
      combined.insert(combined.end(), visited.begin(), visited.end());
    }
    std::sort(combined.begin(), combined.end());
    auto expected = complete;
    std::sort(expected.begin(), expected.end());
    assert(combined == expected);
  }
  // 65536^2 work items: an unsigned-32 product would silently become zero.
  std::vector<std::uint32_t> large(2 * 65536);
  for (std::size_t index = 0; index < large.size(); ++index)
    large[index] = static_cast<std::uint32_t>(index);
  const std::uint32_t large_offsets[]{0, 65536, 65536, 131072};
  run<Unrestricted>({1, large_offsets, large.data()}, 0, 1, nullptr, maximum, maximum);
  assert(visited.size() == 1 && visited.back().first == 131071 && visited.back().second == 65535);
  // The same-class triangular domain also exceeds INT_MAX without wrapping.
  constexpr unsigned triangular_last = 65536ULL * 65537ULL / 2 - 1;
  run<Unrestricted>({1, large_offsets, large.data()}, 0, 0, nullptr,
                    triangular_last, maximum);
  assert(visited.size() == 1 && visited.back().first == 65535 && visited.back().second == 65535);
}
int main() { check<false>(); check<true>(); }
"""
