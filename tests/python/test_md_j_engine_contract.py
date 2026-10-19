"""Execute the SDK-free MD-J hierarchy, source inventory and workspace contract.

These host probes test the actual shared header under both supported language
standards. They qualify integer/screening/scheduling contracts, not GPU numerical
parity, resource occupancy, or complete-endpoint performance.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_SOURCE = r"""
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <vector>
#include "scf/cuda/md_j_engine_contract.hpp"
using namespace generativeqc::scf::cuda_execution;
static_assert(std::is_trivially_default_constructible_v<MdJTileTask>);
using U = std::uint64_t;
using Wide = unsigned __int128;
constexpr U maximum = std::numeric_limits<U>::max();
constexpr auto constant_layout = md_j_hierarchy_layout(33);
static_assert(constant_layout.valid && constant_layout.aligned == 64);
static_assert(constant_layout.entries == 126 && constant_layout.offsets[5] == 124);
constexpr auto constant_schedule = md_j_tile_schedule(4, 4);
static_assert(constant_schedule.valid && constant_schedule.shared_bytes <= 49152);
constexpr auto constant_inventory = md_j_tile_inventory(65, 65, 32, 32, true);
static_assert(constant_inventory.valid && constant_inventory.tasks == 6);

Wide triangle(U n) { return Wide(n) * (Wide(n) + 1) / 2; }
unsigned hermites(unsigned order) { return (order + 1) * (order + 2) * (order + 3) / 6; }

void layouts() {
  for (U count = 0; count <= 1030; ++count) {
    const auto layout = md_j_hierarchy_layout(count);
    assert(layout.valid && layout.count == count);
    assert(layout.aligned == ((count + 31) / 32) * 32);
    assert(layout.entries == 63 * (layout.aligned / 32));
    U offset = 0;
    for (unsigned level = 0; level < 6; ++level) {
      const unsigned width = 1U << level;
      assert(md_j_hierarchy_width(level) == width);
      assert(md_j_hierarchy_level(width) == level);
      assert(layout.offsets[level] == offset);
      for (U group = 0; group < layout.aligned / width; ++group) {
        U index = maximum;
        assert(md_j_hierarchy_index(layout, width, group, index));
        assert(index == offset + group && index < layout.entries);
      }
      U index = maximum;
      assert(!md_j_hierarchy_index(layout, width, layout.aligned / width, index));
      assert(index == 0);
      for (U leaf = 0; leaf < count; ++leaf) {
        assert(md_j_hierarchy_leaf_index(layout, width, leaf, index));
        assert(index == offset + leaf / width);
      }
      assert(!md_j_hierarchy_leaf_index(layout, width, count, index));
      offset += layout.aligned / width;
    }
    assert(offset == layout.entries);
  }
  for (unsigned width : {0U, 3U, 5U, 31U, 33U, 64U}) {
    U index = maximum;
    assert(md_j_hierarchy_level(width) == 6);
    assert(!md_j_hierarchy_index(constant_layout, width, 0, index));
    assert(index == 0);
  }
  assert(md_j_hierarchy_width(6) == 0 && md_j_hierarchy_width(999) == 0);
  const U largest_blocks = maximum / 63;
  const auto largest = md_j_hierarchy_layout(largest_blocks * 32);
  assert(largest.valid && largest.entries == largest_blocks * 63);
  assert(!md_j_hierarchy_layout(largest_blocks * 32 + 1).valid);
  for (U count : {maximum, maximum - 1, maximum - 30, maximum - 31})
    assert(!md_j_hierarchy_layout(count).valid);
  U result = 99;
  assert(!md_j_checked_add(maximum, 1, result) && result == 0);
  assert(!md_j_checked_mul(maximum, 2, result) && result == 0);
  assert(md_j_checked_mul(maximum, 0, result) && result == 0);
  assert(md_j_checked_add(maximum, 0, result) && result == maximum);
}

void storage() {
  U counts[5]{0, 1, 32, 33, 65};
  const auto storage = md_j_hierarchy_storage(counts);
  assert(storage.valid);
  U offset = 0;
  for (unsigned bucket = 0; bucket < 5; ++bucket) {
    assert(storage.buckets[bucket].valid);
    assert(storage.bucket_offsets[bucket] == offset);
    offset += 63 * ((counts[bucket] + 31) / 32);
  }
  assert(storage.bucket_offsets[5] == offset && storage.entries == offset);
  assert(storage.bytes == offset * sizeof(MdJScreenBound));
  assert(md_j_hierarchy_storage(counts, storage.bytes).valid);
  const auto short_storage = md_j_hierarchy_storage(counts, storage.bytes - 1);
  assert(!short_storage.valid && short_storage.entries == 0 && short_storage.bytes == 0);
  assert(!md_j_hierarchy_storage(nullptr).valid);
  U empty[5]{};
  assert(md_j_hierarchy_storage(empty, 0).valid);
  U bad[5]{maximum, 0, 0, 0, 0};
  assert(!md_j_hierarchy_storage(bad).valid);
  // Each hierarchy fits, but the combined allocation/byte count must fail.
  const U large = (maximum / 63) * 32;
  U overflowing[5]{large, large, large, large, large};
  assert(!md_j_hierarchy_storage(overflowing).valid);
  U bytes_overflow[5]{large, 0, 0, 0, 0};
  assert(!md_j_hierarchy_storage(bytes_overflow).valid);
}

void schedules() {
  unsigned pairs = 0;
  for (unsigned bra = 0; bra < 5; ++bra) for (unsigned ket = bra; ket < 5; ++ket) {
    ++pairs;
    const auto schedule = md_j_tile_schedule(bra, ket);
    const auto reverse = md_j_tile_schedule(ket, bra);
    assert(schedule.valid && reverse.valid);
    assert(schedule.threads == 128 && schedule.threads % 32 == 0);
    assert(schedule.bra_width == schedule.ket_width);
    assert(md_j_hierarchy_level(schedule.bra_width) < 6);
    const U expected = 4 * hermites(bra + ket) * sizeof(double) +
        2 * sizeof(double) * (schedule.bra_width * hermites(bra) +
                             schedule.ket_width * hermites(ket)) +
        96 * (schedule.bra_width + schedule.ket_width) + 128;
    assert(schedule.shared_bytes == expected && expected <= 48 * 1024);
    assert(reverse.shared_bytes == schedule.shared_bytes);
    assert(reverse.bra_width == schedule.ket_width);
    for (unsigned width : {1U, 2U, 4U, 8U, 16U, 32U}) {
      U capacity = 0;
      assert(md_j_tile_shared_bytes(bra, ket, width, width, capacity));
      const auto bounded = md_j_tile_schedule(bra, ket, capacity);
      assert(bounded.valid && bounded.shared_bytes <= capacity);
      if (capacity <= kMdJSharedOccupancyTarget) assert(bounded.bra_width == width);
    }
    U minimum = 0;
    assert(md_j_tile_shared_bytes(bra, ket, 1, 1, minimum));
    assert(!md_j_tile_schedule(bra, ket, minimum - 1).valid);
    assert(!md_j_tile_schedule(bra, ket, 0).valid);
  }
  assert(pairs == 15);
  assert(!md_j_tile_schedule(5, 0).valid && !md_j_tile_schedule(0, 5).valid);
  U bytes = 99;
  assert(!md_j_tile_shared_bytes(0, 0, 3, 8, bytes) && bytes == 0);
}

void check_inventory(U bra, U ket, unsigned bw, unsigned kw, bool triangular) {
  const auto inventory = md_j_tile_inventory(bra, ket, bw, kw, triangular);
  assert(inventory.valid);
  const U btiles = bra / bw + (bra % bw != 0);
  const U ktiles = ket / kw + (ket % kw != 0);
  const U expected = triangular ? U(triangle(btiles)) : btiles * ktiles;
  assert(inventory.tasks == expected);
  std::vector<unsigned> visits(bra * ket, 0);
  std::vector<unsigned> tile_visits(btiles * ktiles, 0);
  for (U index = 0; index < inventory.tasks; ++index) {
    MdJTileTask task{};
    assert(md_j_decode_tile(inventory, index, task));
    assert(task.bra_begin == task.bra_tile * bw && task.ket_begin == task.ket_tile * kw);
    assert(task.bra_count && task.bra_count <= bw && task.ket_count && task.ket_count <= kw);
    assert(task.bra_begin + task.bra_count <= bra && task.ket_begin + task.ket_count <= ket);
    const U encoded = triangular ? U(triangle(task.ket_tile)) + task.bra_tile
                                : task.bra_tile * ktiles + task.ket_tile;
    assert(encoded == index);
    assert(++tile_visits[task.bra_tile * ktiles + task.ket_tile] == 1);
    for (unsigned b = 0; b < bw; ++b) for (unsigned k = 0; k < kw; ++k) {
      const bool wanted = task.bra_begin + b < bra && task.ket_begin + k < ket &&
          (!triangular || task.bra_begin + b <= task.ket_begin + k);
      assert(md_j_tile_source_valid(inventory, task, b, k) == wanted);
      if (wanted) assert(++visits[(task.bra_begin + b) * ket + task.ket_begin + k] == 1);
    }
  }
  for (U b = 0; b < bra; ++b) for (U k = 0; k < ket; ++k)
    assert(visits[b * ket + k] == unsigned(!triangular || b <= k));
  MdJTileTask bad{1, 1, 1, 1, 1, 1};
  assert(!md_j_decode_tile(inventory, inventory.tasks, bad));
  assert(!bad.bra_count && !bad.ket_count && !bad.bra_begin && !bad.ket_begin);
  assert(!md_j_decode_tile(inventory, maximum, bad));
}

void inventories() {
  for (U bra : {0U, 1U, 7U, 8U, 15U, 16U, 31U, 32U, 33U, 65U}) {
    for (unsigned bw : {1U, 2U, 4U, 8U, 16U, 32U}) {
      check_inventory(bra, bra, bw, bw, true);
      for (U ket : {0U, 1U, 9U, 32U, 35U})
        for (unsigned kw : {1U, 2U, 4U, 8U, 16U, 32U})
          check_inventory(bra, ket, bw, kw, false);
    }
  }
  assert(!md_j_tile_inventory(1, 2, 8, 8, true).valid);
  assert(!md_j_tile_inventory(2, 2, 8, 4, true).valid);
  assert(!md_j_tile_inventory(2, 2, 0, 8, false).valid);
  assert(!md_j_tile_inventory(2, 2, 8, 3, false).valid);
  assert(!md_j_tile_inventory(maximum, maximum, 1, 1, false).valid);
}

void check_large(const MdJTileInventory& inventory, U index) {
  assert(inventory.valid && index < inventory.tasks);
  MdJTileTask task{};
  assert(md_j_decode_tile(inventory, index, task));
  if (inventory.triangular) {
    assert(task.bra_tile <= task.ket_tile);
    assert(triangle(task.ket_tile) <= index);
    assert(triangle(task.ket_tile + 1) > index);
    assert(triangle(task.ket_tile) + task.bra_tile == index);
  } else {
    assert(Wide(task.bra_tile) * inventory.ket_tiles + task.ket_tile == index);
  }
  assert(Wide(task.bra_begin) + task.bra_count <= inventory.bra_count);
  assert(Wide(task.ket_begin) + task.ket_count <= inventory.ket_count);
}

void large_boundaries() {
  U low = 0, high = maximum;
  while (low < high) {
    const U middle = low + (high - low) / 2 + 1;
    if (triangle(middle) <= maximum) low = middle; else high = middle - 1;
  }
  const U largest_n = low;
  U tasks = 0;
  assert(md_j_triangular_count(largest_n, tasks) && Wide(tasks) == triangle(largest_n));
  assert(!md_j_triangular_count(largest_n + 1, tasks) && tasks == 0);
  assert(!md_j_triangular_count(maximum, tasks) && tasks == 0);
  const auto largest = md_j_tile_inventory(largest_n, largest_n, 1, 1, true);
  assert(largest.valid);
  assert(!md_j_tile_inventory(largest_n + 1, largest_n + 1, 1, 1, true).valid);
  for (U row : {U(1), U(2), U(134217727), U(134217728), U(4294967295),
                largest_n - 2, largest_n - 1}) {
    const U start = U(triangle(row));
    check_large(largest, start - 1);
    check_large(largest, start);
    check_large(largest, start + 1);
    check_large(largest, U(triangle(row + 1)) - 1);
  }
  U random = 0x725641302415abcdULL;
  for (unsigned sample = 0; sample < 20000; ++sample) {
    random = random * 6364136223846793005ULL + 1442695040888963407ULL;
    check_large(largest, random % largest.tasks);
  }
  const auto scaled = md_j_tile_inventory(largest_n * 32, largest_n * 32, 32, 32, true);
  check_large(scaled, scaled.tasks - 1);
  for (unsigned width : {1U, 2U, 4U, 8U, 16U, 32U}) {
    const auto rectangle = md_j_tile_inventory(maximum, 1, width, 1, false);
    for (U index : {U(0), U(1), rectangle.tasks / 2, rectangle.tasks - 1})
      check_large(rectangle, index);
    MdJTileTask tail{};
    assert(md_j_decode_tile(rectangle, rectangle.tasks - 1, tail));
    assert(tail.bra_count == (maximum % width ? maximum % width : width));
  }
}

void bounds() {
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double tiny = std::numeric_limits<double>::denorm_min();
  const double huge = std::numeric_limits<double>::max();
  for (double first : {0.0, tiny, 1e-200, 0.1, 1.0, 1e200, huge})
    for (double second : {0.0, tiny, 1e-200, 0.1, 1.0, 1e200, huge}) {
      const double bound = md_j_bound_product_up(first, second);
      assert(bound >= 0.0);
      assert(static_cast<long double>(bound) >=
             static_cast<long double>(first) * static_cast<long double>(second));
      if (first == 0.0 || second == 0.0) assert(bound == 0.0);
    }
  assert(md_j_bound_product_up(tiny, tiny) >= tiny);
  for (double invalid : {-1.0, -inf, nan}) {
    assert(md_j_nonnegative_bound(invalid) == inf);
    assert(md_j_bound_product_up(invalid, 0.0) == inf);
    assert(md_j_bound_product_up(0.0, invalid) == inf);
  }
  const MdJScreenBound leaves[]{md_j_screen_bound(0.0, 0.0, 0.0),
      md_j_screen_bound(2.0, 0.125, 0.5), md_j_screen_bound(0.25, 17.0, 0.0625),
      md_j_screen_bound(8.0, 0.03125, 2.0), md_j_screen_bound(1e-200, 1e200, 1e-201)};
  MdJScreenBound root{};
  for (const auto leaf : leaves) root = md_j_merge_screen_bounds(root, leaf);
  assert(root.q == 8.0 && root.minimum == 2.0 && root.density == 1e200);
  const auto left = md_j_merge_screen_bounds(leaves[0], leaves[1]);
  const auto right = md_j_merge_screen_bounds(leaves[2], leaves[3]);
  const auto tree = md_j_merge_screen_bounds(md_j_merge_screen_bounds(left, right), leaves[4]);
  assert(tree.q == root.q && tree.qd == root.qd && tree.minimum == root.minimum &&
         tree.density == root.density);
  for (const auto first : leaves) for (const auto second : leaves) {
    assert(root.q >= first.q && root.qd >= first.qd && root.minimum >= first.minimum);
    assert(root.density >= first.density);
    assert(md_j_screen_density_bound(root, root) >= md_j_screen_density_bound(first, second));
    assert(md_j_bound_product_up(root.minimum, root.minimum) >=
           md_j_bound_product_up(first.minimum, second.minimum));
  }
  // Keep multiplication order: a cached q*d is not a substitute for (q*q)*d.
  const auto output = md_j_screen_bound(1e308, 0.0, 0.0);
  const auto input = md_j_screen_bound(1e-308, 1e-308, 0.0);
  assert(md_j_screen_density_bound(output, input) > 1e-310);
  assert(md_j_bound_product_up(output.q, input.qd) > 1e-17);
  const auto poisoned = md_j_merge_screen_bounds(root, {nan, nan, nan, nan});
  assert(poisoned.q == inf && poisoned.qd == inf && poisoned.minimum == inf &&
         poisoned.density == inf);
}

int main(int argc, char** argv) {
  assert(argc == 2);
  switch (std::atoi(argv[1])) {
    case 0: layouts(); break;
    case 1: storage(); break;
    case 2: schedules(); break;
    case 3: inventories(); break;
    case 4: large_boundaries(); break;
    case 5: bounds(); break;
    default: assert(false);
  }
}
"""


@pytest.fixture(scope="module", params=["c++17", "c++20"])
def engine_probe(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Verify and use the preferred available compiler cache for the real header."""
    compiler = shutil.which("c++")
    cache = shutil.which("sccache") or shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("requires a C++ compiler and sccache or ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True, timeout=10)
    directory = tmp_path_factory.mktemp("md_j_engine_contract")
    source, obj, binary = (
        directory / "probe.cpp",
        directory / "probe.o",
        directory / "probe",
    )
    source.write_text(_SOURCE)
    subprocess.run(
        [
            cache,
            compiler,
            f"-std={request.param}",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT / "src"),
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
    )
    subprocess.run(
        [compiler, str(obj), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return binary


@pytest.mark.parametrize(
    "section",
    range(6),
    ids=[
        "six_level_offsets",
        "five_bucket_capacity",
        "all_angular_schedules",
        "reciprocal_bijection_and_tails",
        "uint64_triangular_boundaries",
        "conservative_bounds",
    ],
)
def test_md_j_engine_contract(engine_probe: Path, section: int) -> None:
    subprocess.run([str(engine_probe), str(section)], check=True, timeout=30)
