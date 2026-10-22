"""Execute the real MD-J CUDA kernel bodies with synchronized host CUDA doubles.

This tests native scheduling, shared-cache publication, reciprocal accumulation,
packed pair locations and hierarchy semantics. It is not a GPU race/sanitizer or
performance qualification. The generated serial contraction is the scheduling
oracle; test_md_j_engine_math.py independently qualifies its numerical science.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from _cpp_source_support import cpp_function_definition, cpp_record_definition
from generativeqc_compiler.integral.md_j_engine_cuda import emit_md_j_engine_header

ROOT = Path(__file__).resolve().parents[2]


def _sanitizer_environment() -> dict[str, str]:
    # LeakSanitizer cannot run under some CI process tracers. This probe checks
    # address validity and undefined behavior; separate owner tests cover leaks.
    return {
        **os.environ,
        "ASAN_OPTIONS": os.environ.get("ASAN_OPTIONS", "") + ":detect_leaks=0",
    }


@pytest.fixture(
    scope="module",
    params=["undefined", "address,undefined"],
    ids=["ubsan", "asan_ubsan"],
)
def engine_runtime_probe(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    requested_cache = os.environ.get("CCACHE")
    launcher = (
        shutil.which(requested_cache)
        if requested_cache
        else shutil.which("sccache") or shutil.which("ccache")
    )
    if not compiler or not launcher:
        pytest.skip("requires a C++20 compiler and verified sccache/ccache")
    subprocess.run([launcher, "--version"], check=True, capture_output=True, timeout=10)
    folder = tmp_path_factory.mktemp("md-j-engine-runtime")
    sanitizers = request.param
    use_asan = "address" in sanitizers
    if use_asan:
        # Some supported host toolchains lack ASan or its instrumentation API.
        # Qualify availability with a tiny cached compile/link/run, while always
        # retaining the separate UBSan and guard-sentinel execution variant.
        check_source = folder / "asan-check.cpp"
        check_object, check_binary = folder / "asan-check.o", folder / "asan-check"
        check_source.write_text(
            "#include <sanitizer/asan_interface.h>\n"
            "int main() { int value = 0; "
            "__asan_unpoison_memory_region(&value, sizeof(value)); return value; }\n"
        )
        try:
            subprocess.run(
                [
                    launcher,
                    compiler,
                    "-fsanitize=address",
                    "-c",
                    str(check_source),
                    "-o",
                    str(check_object),
                ],
                check=True,
                capture_output=True,
                timeout=30,
                env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
            )
            subprocess.run(
                [
                    compiler,
                    "-fsanitize=address",
                    str(check_object),
                    "-o",
                    str(check_binary),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            subprocess.run(
                [str(check_binary)],
                check=True,
                capture_output=True,
                timeout=10,
                env=_sanitizer_environment(),
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pytest.skip("optional AddressSanitizer instrumentation is unavailable")
    (folder / "cuda_runtime_api.h").write_text(
        "#pragma once\nusing cudaStream_t = void*;\nusing cudaError_t = int;\n"
    )
    (folder / "generated_md_j_engine.cuh").write_text(emit_md_j_engine_header())
    native = (ROOT / "src/scf/cuda/md_j_engine.cuh").read_text()
    kernels = (
        cpp_record_definition(native, "MdJEngineCachedPair")
        + ";\n"
        + "\n".join(
            cpp_function_definition(native, name, include_template=templated)
            for name, templated in (
                ("md_j_primitive_allowance", False),
                ("md_j_engine_pair_loc", False),
                ("md_j_engine_source_threshold", False),
                ("md_j_engine_count_source_norm", False),
                ("md_j_engine_source_tails", False),
                ("md_j_engine_tail_threshold", False),
                ("md_j_engine_hierarchy", False),
                ("md_j_engine_direction", False),
                ("md_j_engine_primitive_direction", False),
                ("md_j_engine_tile_admitted", False),
                ("md_j_engine_cached_pair", False),
                ("md_j_engine_resident_blocks", True),
                ("md_j_engine_potential", True),
            )
        )
    )
    source, obj, binary = folder / "probe.cpp", folder / "probe.o", folder / "probe"
    source.write_text(_CUDA_DOUBLES + kernels + _DRIVER)
    subprocess.run(
        [
            launcher,
            compiler,
            "-std=c++20",
            "-O1",
            "-g",
            "-ffp-contract=off",
            f"-fsanitize={sanitizers}",
            f"-DMD_J_HOST_ASAN={int(use_asan)}",
            "-fno-sanitize-recover=all",
            "-pthread",
            "-I" + str(folder),
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
    )
    subprocess.run(
        [compiler, "-pthread", f"-fsanitize={sanitizers}", str(obj), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return binary


@pytest.mark.parametrize(
    "section",
    range(8),
    ids=(
        "packed_locations_and_all_hierarchy_levels",
        "all_fifteen_angular_pairs",
        "ragged_tiles_and_same_pair_reciprocity",
        "asymmetric_density_masks_and_windows",
        "rejected_tiles_and_uncounted_replay",
        "exact_leaf_and_tile_screening_boundaries",
        "primitive_threshold_hierarchy_and_replay",
        "source_tail_cumulative_budget",
    ),
)
def test_native_md_j_engine_host_semantics(
    engine_runtime_probe: Path, section: int
) -> None:
    # A broken collective must fail finitely instead of wedging the test runner.
    subprocess.run(
        [str(engine_runtime_probe), str(section)],
        check=True,
        timeout=120,
        env=_sanitizer_environment(),
    )


_CUDA_DOUBLES = r"""
#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <thread>
#include <vector>
#if MD_J_HOST_ASAN
#include <sanitizer/asan_interface.h>
#endif
#define __host__
#define __device__
#define __global__
#define __launch_bounds__(...)
#define __shared__ static
#define __constant__
#define __forceinline__ inline
#include "generated_md_j_engine.cuh"
using namespace generativeqc::scf::cuda_execution;

struct Dim { unsigned x; };
thread_local Dim threadIdx{};
Dim blockIdx{};
constexpr Dim blockDim{128};
std::barrier<> block_barrier(128);
std::barrier<> warp0(32), warp1(32), warp2(32), warp3(32);
std::barrier<>* warp_barriers[]{&warp0, &warp1, &warp2, &warp3};
std::uint64_t shuffled[128]{};
void __syncthreads() {
  block_barrier.arrive_and_wait();
  // Legal scheduling skew exposes control-state reuse without adding a barrier.
  if (threadIdx.x == 127) for (unsigned i = 0; i < 8; ++i) std::this_thread::yield();
}
void __syncwarp(unsigned mask = 0xffffffffU) {
  assert(mask == 0xffffffffU);
  warp_barriers[threadIdx.x / 32]->arrive_and_wait();
}
template <class T>
T __shfl_down_sync(unsigned mask, T value, unsigned delta, unsigned width = 32) {
  static_assert(sizeof(T) <= sizeof(std::uint64_t));
  assert(mask == 0xffffffffU && width && width <= 32 && (width & (width - 1)) == 0);
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(value));
  shuffled[threadIdx.x] = bits;
  __syncwarp(mask);
  const unsigned lane = threadIdx.x % width;
  const unsigned source = lane + delta < width ? threadIdx.x + delta : threadIdx.x;
  bits = shuffled[source];
  __syncwarp(mask);
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}
template <class T> T atomicAdd(T* address, T value) {
  return std::atomic_ref<T>(*address).fetch_add(value);
}
template <class T> T atomicMax(T* address, T value) {
  std::atomic_ref<T> target(*address);
  T previous = target.load();
  while (previous < value && !target.compare_exchange_weak(previous, value)) {}
  return previous;
}
long long __double_as_longlong(double value) {
  long long bits; std::memcpy(&bits, &value, sizeof(bits)); return bits;
}
double __longlong_as_double(long long bits) {
  double value; std::memcpy(&value, &bits, sizeof(value)); return value;
}
double __dadd_ru(double a, double b) {
  const long double exact = static_cast<long double>(a) + b;
  const double rounded = exact;
  return rounded < exact ? std::nextafter(rounded, INFINITY) : rounded;
}
double __dmul_ru(double a, double b) {
  const long double exact = static_cast<long double>(a) * b;
  const double rounded = exact;
  return rounded < exact ? std::nextafter(rounded, INFINITY) : rounded;
}
double __ddiv_rd(double a, double b) {
  const long double exact = static_cast<long double>(a) / b;
  const double rounded = exact;
  return rounded > exact ? std::nextafter(rounded, -INFINITY) : rounded;
}

// Reuse all 128 host workers across launches. Every warp has all 32 lanes;
// emulation is deliberately one block at a time because CUDA shared variables
// become function-local statics, not independently allocated block storage.
class Block {
  std::barrier<> start{129}, finish{129};
  std::array<std::thread, 128> workers;
  std::function<void()> body;
  bool stopping = false;
 public:
  Block() {
    for (unsigned t = 0; t < workers.size(); ++t) workers[t] = std::thread([&, t] {
      threadIdx.x = t;
      for (;;) {
        start.arrive_and_wait();
        if (stopping) return;
        body();
        finish.arrive_and_wait();
      }
    });
  }
  ~Block() {
    stopping = true;
    start.arrive_and_wait();
    for (auto& worker : workers) worker.join();
  }
  void run(std::function<void()> function, unsigned index = 0) {
    body = std::move(function);
    blockIdx.x = index;
    start.arrive_and_wait();
    finish.arrive_and_wait();
  }
};
"""

_DRIVER = r"""

constexpr double sentinel = -1.23456789012345e150;
template <class T> class Guarded {
  std::vector<T> values;
  std::array<unsigned char, sizeof(T)> first_guard{}, last_guard{};
 public:
  Guarded(std::size_t count, T value) : values(count + 2, value) {
    std::memcpy(first_guard.data(), &values.front(), sizeof(T));
    std::memcpy(last_guard.data(), &values.back(), sizeof(T));
  }
  T* data() { return values.data() + 1; }
  const T* data() const { return values.data() + 1; }
  T& operator[](std::size_t index) { assert(index < size()); return data()[index]; }
  const T& operator[](std::size_t index) const { assert(index < size()); return data()[index]; }
  std::size_t size() const { return values.size() - 2; }
  void check() const {
    assert(std::memcmp(&values.front(), first_guard.data(), sizeof(T)) == 0);
    assert(std::memcmp(&values.back(), last_guard.data(), sizeof(T)) == 0);
  }
};

struct Fixture {
  std::array<std::uint64_t, 5> sizes;
  MdJView md{};
  Guarded<MdJPair> pairs;
  Guarded<MdJPrimitive> primitives;
  Guarded<std::uint32_t> ordered;
  Guarded<double> minimum, maximum, bounds, density, potential;
  Guarded<MdJScreenBound> hierarchy;
  Guarded<MdJWorkCounts> census{1, {}};
  unsigned long long cursor = 0;
  unsigned mode;
  std::vector<MdJPrimitive> original;
  std::vector<double> original_density;

  static std::size_t sum(const std::array<std::uint64_t, 5>& n) {
    return std::accumulate(n.begin(), n.end(), std::size_t(0));
  }
  static std::size_t pair_size(const std::array<std::uint64_t, 5>& n) {
    std::size_t out = 0; for (auto count : n) out += (count + 1) / 2; return out;
  }
  static std::size_t hermites(const std::array<std::uint64_t, 5>& n) {
    std::size_t out = 0;
    for (unsigned a = 0; a < 5; ++a) out += n[a] * md_j_hermite_count(a);
    return out;
  }
  Fixture(std::array<std::uint64_t, 5> n, unsigned pattern = 0)
      : sizes(n), pairs(pair_size(n), {}), primitives(sum(n), {}),
        ordered(sum(n), 0xffffffffU), minimum(pair_size(n), sentinel),
        maximum(pair_size(n), sentinel), bounds(pair_size(n), sentinel),
        density(hermites(n), sentinel), potential(hermites(n), sentinel),
        hierarchy(md_j_hierarchy_storage(n.data()).entries,
                  {sentinel, sentinel, sentinel, sentinel}), mode(pattern) {
    md.pair_count = pairs.size(); md.primitive_count = primitives.size();
    md.pairs = pairs.data(); md.primitives = primitives.data();
    md.ordered_primitives = ordered.data(); md.minimum_bounds = minimum.data();
    md.maximum_bounds = maximum.data(); md.density_bounds = bounds.data();
    md.density = density.data(); md.potential = potential.data();
    md.hierarchy_layout = md_j_hierarchy_storage(n.data());
    assert(md.hierarchy_layout.valid);
    md.hierarchy = hierarchy.data(); md.source_cursor = &cursor;
    md.work_counts = census.data();
    unsigned p = 0, next = 0;
    for (unsigned a = 0; a < 5; ++a) {
      md.class_offsets[a] = next;
      md.pair_loc_offsets[a + 1] = md.pair_loc_offsets[a] + n[a] * md_j_hermite_count(a);
      for (unsigned j = 0; j < n[a]; ++j) {
        if (j % 2 == 0) {
          if (j) ++p;
          pairs[p] = {};
          pairs[p].angular = a;
          pairs[p].system = mode == 4 ? (j / 2) % 3 : 0;
          minimum[p] = mode == 1 && (j / 2) % 4 == 0 ? 0.125 : 1.0;
          maximum[p] = 2.0 + (n[a] - j) * 0.125;
          bounds[p] = mode == 3 ? 0.0 :
              mode == 1 && (j / 2) % 5 == 0 ? 0.0 :
              mode == 1 && (j / 2) % 5 == 1 ? 1e-20 : 0.25 + 0.03125 * (1 + j % 7);
          // Highest-q primitives are sorted first. This creates rejected tiles
          // immediately before accepted ones to test shared control lifetimes.
          if (mode == 5 && j < 32) bounds[p] = 0.0;
        }
        // Physical IDs run backwards across classes, unlike sorted tile order.
        const unsigned id = primitives.size() - 1 - next;
        auto& primitive = primitives[id];
        primitive = {};
        primitive.pair = p;
        primitive.first = 1000 + id;
        primitive.second = 2000 + 3 * id;
        primitive.transform_offset = 17 + 5 * id;
        primitive.hermite_offset = std::numeric_limits<std::size_t>::max();
        primitive.exponent = 0.25 + (1 + id % 13) * 0.0625;
        primitive.product = {0.03 * (int(id % 7) - 3),
                             -0.05 * (int(id % 11) - 5),
                             0.07 * (int(id % 5) - 2)};
        ordered[next++] = id;
      }
      if (n[a]) ++p;
      auto* first = ordered.data() + md.class_offsets[a];
      std::sort(first, ordered.data() + next, [&](unsigned x, unsigned y) {
        const auto qx = maximum[primitives[x].pair], qy = maximum[primitives[y].pair];
        return qx > qy || (qx == qy && x < y);
      });
    }
    md.class_offsets[5] = next;
    assert(p == pairs.size() && next == primitives.size());
    original.assign(primitives.data(), primitives.data() + primitives.size());
  }
  void pack(Block& block) {
    for (unsigned angular = 0; angular < 5; ++angular) {
      const unsigned count = sizes[angular];
      for (unsigned i = 0; i < (count + 127) / 128; ++i)
        block.run([&, angular] { md_j_engine_pair_loc(md, angular); }, i);
    }
    std::vector<unsigned> seen(density.size(), 0);
    for (unsigned a = 0; a < 5; ++a) for (unsigned i = 0; i < sizes[a]; ++i) {
      const auto id = ordered[md.class_offsets[a] + i];
      const auto& primitive = primitives[id];
      const auto offset = md.pair_loc_offsets[a] + i * md_j_hermite_count(a);
      assert(primitive.hermite_offset == offset);
      const auto& expected = original[id];
      assert(primitive.pair == expected.pair && primitive.first == expected.first &&
             primitive.second == expected.second && primitive.transform_offset == expected.transform_offset);
      assert(primitive.exponent == expected.exponent && primitive.product.x == expected.product.x &&
             primitive.product.y == expected.product.y && primitive.product.z == expected.product.z);
      for (unsigned c = 0; c < md_j_hermite_count(a); ++c) {
        assert(offset + c < seen.size() && ++seen[offset + c] == 1);
        density[offset + c] = bounds[primitive.pair] == 0.0 ? 0.0 :
            bounds[primitive.pair] * (int((id + 3 * c) % 11) - 5) / 7.0;
        potential[offset + c] = 0.0;
      }
    }
    assert(std::all_of(seen.begin(), seen.end(), [](unsigned n) { return n == 1; }));
    original_density.assign(density.data(), density.data() + density.size());
    check();
  }
  void build(Block& block, unsigned begin, unsigned end) {
    for (unsigned a = 0; a < 5; ++a) {
      const auto& layout = md.hierarchy_layout.buckets[a];
      for (unsigned i = 0; i < (layout.aligned + 127) / 128; ++i)
        block.run([&, a] { md_j_engine_hierarchy(md, a, begin, end); }, i);
      for (unsigned level = 0; level < 6; ++level) {
        const unsigned width = 1U << level;
        for (unsigned group = 0; group < layout.aligned / width; ++group) {
          MdJScreenBound expected{};
          for (unsigned j = group * width; j < (group + 1) * width && j < layout.count; ++j) {
            const auto pair = primitives[ordered[md.class_offsets[a] + j]].pair;
            const auto system = unsigned(pairs[pair].system);
            if (system < begin || system >= end) continue;
            expected.q = std::max(expected.q, maximum[pair]);
            // The SDK-free bound helper deliberately expands finite products
            // by one ULP. This is more conservative than CUDA directed rounding.
            const double qd = maximum[pair] == 0 || bounds[pair] == 0 ? 0.0 :
                std::nextafter(maximum[pair] * bounds[pair], INFINITY);
            expected.qd = std::max(expected.qd, qd);
            expected.minimum = std::max(expected.minimum, minimum[pair]);
            expected.density = std::max(expected.density, bounds[pair]);
          }
          const auto actual = hierarchy[md.hierarchy_layout.bucket_offsets[a] +
                                        layout.offsets[level] + group];
          assert(actual.q == expected.q && actual.qd == expected.qd);
          assert(actual.minimum == expected.minimum && actual.density == expected.density);
        }
      }
    }
    check();
  }
  void poison_excluded_density(unsigned begin, unsigned end) {
#if MD_J_HOST_ASAN
    // Unrequested systems are not initialized by the density transform on a
    // first subset replay. Mark those reads invalid, plus the inactive suffix.
    // This models availability with host ASan; it is not GPU initcheck evidence.
    for (unsigned a = 0; a < 5; ++a) for (unsigned i = 0; i < sizes[a]; ++i) {
      const auto& primitive = primitives[ordered[md.class_offsets[a] + i]];
      const auto system = unsigned(pairs[primitive.pair].system);
      if (system < begin || system >= end || i >= md.hierarchy_layout.buckets[a].count)
        __asan_poison_memory_region(md.density + primitive.hermite_offset,
                                    md_j_hermite_count(a) * sizeof(double));
    }
#else
    (void)begin; (void)end;
#endif
  }
  void restore_density_access() {
#if MD_J_HOST_ASAN
    __asan_unpoison_memory_region(md.density, density.size() * sizeof(double));
#endif
  }
  void check() const {
    pairs.check(); primitives.check(); ordered.check(); minimum.check(); maximum.check();
    bounds.check(); density.check(); potential.check(); hierarchy.check(); census.check();
  }
};

bool allowed(const Fixture& f, unsigned output, unsigned input, double screening) {
  const double allowance = __ddiv_rd(std::min(screening, 1e-12), double(f.md.pair_count));
  const double contribution = __dmul_ru(__dmul_ru(f.maximum[output], f.maximum[input]),
                                        f.bounds[input]);
  return f.minimum[output] * f.minimum[input] >= screening &&
         f.bounds[input] != 0.0 && contribution >= allowance;
}

unsigned completed = 0;
void screening_boundaries() {
  // Products/divisions by one and two are exact for these normal binary values,
  // so the host directed-rounding doubles faithfully represent equality here.
  double minimum[]{1.0, 1.0}, maximum[]{1.0, 1.0}, density[]{1.0, 1.0};
  const std::uint64_t sizes[]{1, 1, 0, 0, 0};
  MdJView md{};
  md.pair_count = 2;
  md.minimum_bounds = minimum; md.maximum_bounds = maximum; md.density_bounds = density;
  md.hierarchy_layout = md_j_hierarchy_storage(sizes);
  std::vector<MdJScreenBound> hierarchy(md.hierarchy_layout.entries);
  md.hierarchy = hierarchy.data();
  constexpr auto schedule = md_j_tile_schedule(0, 1);
  const MdJTileTask task{0, 0, 0, 0, 1, 1};
  const auto node = [&](unsigned a) -> MdJScreenBound& {
    const unsigned width = a == 0 ? schedule.bra_width : schedule.ket_width;
    return hierarchy[md.hierarchy_layout.bucket_offsets[a] +
                     md.hierarchy_layout.buckets[a].offsets[md_j_hierarchy_level(width)]];
  };
  const auto refresh = [&] {
    for (unsigned a = 0; a < 2; ++a)
      node(a) = {maximum[a], maximum[a] * density[a], minimum[a], density[a]};
  };
  const auto tile = [&](double screening) {
    return md_j_engine_tile_admitted(md, 0, 1, schedule, task, screening);
  };
  refresh();
  // The public minimum test is nearest-rounded multiplication with >= equality.
  for (double screening : {std::nextafter(1.0, 0.0), 1.0, std::nextafter(1.0, INFINITY)}) {
    const bool accepted = screening <= 1.0;
    assert(md_j_engine_direction(md, 0, 1, screening) == accepted);
    assert(md_j_engine_direction(md, 1, 0, screening) == accepted);
    assert(tile(screening) == accepted);
  }
  for (double value : {std::nextafter(1.0, 0.0), 1.0, std::nextafter(1.0, INFINITY)}) {
    minimum[1] = value; refresh();
    assert(md_j_engine_direction(md, 0, 1, 1.0) == (value >= 1.0));
    assert(tile(1.0) == (value >= 1.0));
  }
  minimum[1] = 1.0;
  const double screening = 0x1p-40;  // Below the density-error cap.
  const double allowance = 0x1p-41;  // Exactly screening / pair_count.
  density[0] = 0.0;
  for (double value : {std::nextafter(allowance, 0.0), allowance,
                       std::nextafter(allowance, INFINITY)}) {
    density[1] = value; refresh();
    assert(md_j_engine_direction(md, 0, 1, screening) == (value >= allowance));
    assert(!md_j_engine_direction(md, 1, 0, screening));
    // Hierarchy products round conservatively outward; they may retain the
    // immediate-below neighbor but must never reject an admitted leaf.
    if (value >= allowance) assert(tile(screening));
  }
  density[1] = allowance / 2; refresh();
  assert(!tile(screening) && !md_j_engine_direction(md, 0, 1, screening));
  density[1] = 0.0; refresh();
  assert(!tile(screening));
  density[0] = allowance; refresh();
  assert(tile(screening));
  assert(md_j_engine_direction(md, 1, 0, screening));
  assert(!md_j_engine_direction(md, 0, 1, screening));

  // The primitive recipe has an absolute cumulative cap, independent of the
  // legacy exact mask above. Probe divisions at normal/subnormal extremes and
  // neighboring source norms against a wider independent product.
  for (double q : {0.0, 0x1p-1022, 0x1p-400, 0.1, 1.0, 0x1p400, 0x1p1023}) {
    for (double share : {0.0, 0x1p-1074, 0x1p-60, 1e-12}) {
      const double threshold = md_j_engine_source_threshold(q, share);
      assert(threshold >= 0.0);
      for (double source : {0.0, std::nextafter(threshold, 0.0), threshold,
                            std::nextafter(threshold, INFINITY)}) {
        if (!md_j_engine_primitive_direction(source, threshold) && source != 0.0)
          assert(static_cast<long double>(q) * source < static_cast<long double>(share));
      }
      if (share == 0.0) assert(threshold == 0.0);
    }
  }
  for (double q : {-1.0, kMdJBoundInfinity, std::numeric_limits<double>::quiet_NaN()})
    assert(md_j_engine_source_threshold(q, 1e-12) == 0.0);
  assert(md_j_engine_primitive_direction(INFINITY,
                                       md_j_engine_source_threshold(0.0, 1e-12)));
  md.maximum_pair_primitives = 81;
  md.primitive_count = 100003;
  for (unsigned targets : {1, 2, 6, 9, 81}) {
    const double share = md_j_primitive_allowance(md, 0.02, targets);
    assert(static_cast<long double>(share) * targets * md.primitive_count <= 1e-12L);
    assert(share >= md_j_primitive_allowance(md, 0.02));
  }
  ++completed;
}
template <unsigned B, unsigned K, bool Count = true>
void evaluate(Block& block, unsigned nb, unsigned nk, unsigned mode = 0,
              unsigned begin = 0, unsigned end = 1, double screening = 0.25) {
  if constexpr (B == K) nk = nb;
  std::array<std::uint64_t, 5> sizes{1, 1, 1, 1, 1};
  sizes[B] = nb; sizes[K] = nk;
  Fixture f(sizes, mode);
  if (mode == 2) {
    // One entire class has no density; for equal angular orders alternate
    // contracted pairs to exercise both independent reciprocal directions.
    for (unsigned p = 0; p < f.pairs.size(); ++p)
      if (f.pairs[p].angular == B && (B != K || p % 2 == 0)) f.bounds[p] = 0.0;
  }
  if (mode == 6) {
    // Packing reserves every primitive, while the hierarchy and tile inventory
    // consume only the eligible sorted prefix. Excluded pairs have an exact
    // zero minimum, so the full serial oracle independently rejects them.
    for (const unsigned a : {B, K}) {
      const auto active = (sizes[a] / 4) * 2;
      f.md.hierarchy_layout.buckets[a].count = active;
      for (unsigned i = active; i < sizes[a]; ++i)
        f.minimum[f.primitives[f.ordered[f.md.class_offsets[a] + i]].pair] = 0.0;
    }
  }
  f.pack(block); f.build(block, begin, end);
  const auto schedule = md_j_tile_schedule(B, K);
  const auto inventory = md_j_tile_inventory(f.md.hierarchy_layout.buckets[B].count,
                                            f.md.hierarchy_layout.buckets[K].count,
                                            schedule.bra_width, schedule.ket_width, B == K);
  assert(inventory.valid);
  std::vector<double> expected(f.potential.size(), 0.0), magnitude(expected.size(), 0.0);
  MdJWorkCounts expected_counts{};
  expected_counts.engine_tiles[5 * B + K] = inventory.tasks;
  const auto product_up = [](double a, double b) {
    return a == 0.0 || b == 0.0 ? 0.0 : std::nextafter(a * b, INFINITY);
  };
  for (unsigned bt = 0; bt < inventory.bra_tiles; ++bt)
    for (unsigned kt = 0; kt < inventory.ket_tiles; ++kt) {
      if (B == K && bt > kt) continue;
      const auto get = [&](unsigned angular, unsigned tile, unsigned width) {
        const auto& layout = f.md.hierarchy_layout.buckets[angular];
        return f.hierarchy[f.md.hierarchy_layout.bucket_offsets[angular] +
                           layout.offsets[md_j_hierarchy_level(width)] + tile];
      };
      const auto bra = get(B, bt, schedule.bra_width);
      const auto ket = get(K, kt, schedule.ket_width);
      const double allowance = __ddiv_rd(std::min(screening, 1e-12), double(f.md.pair_count));
      const double qq = product_up(bra.q, ket.q);
      const bool admitted = bra.minimum * ket.minimum >= screening &&
          ((ket.density != 0.0 && product_up(qq, ket.density) >= allowance) ||
           (bra.density != 0.0 && product_up(qq, bra.density) >= allowance));
      expected_counts.engine_screened_tiles[5 * B + K] += !admitted;
    }
  for (unsigned bi = 0; bi < nb; ++bi) for (unsigned ki = 0; ki < nk; ++ki) {
    if (B == K && bi > ki) continue;
    const auto& bra = f.primitives[f.ordered[f.md.class_offsets[B] + bi]];
    const auto& ket = f.primitives[f.ordered[f.md.class_offsets[K] + ki]];
    const auto system = unsigned(f.pairs[bra.pair].system);
    if (system < begin || system >= end || int(system) != f.pairs[ket.pair].system) continue;
    const bool forward = allowed(f, bra.pair, ket.pair, screening);
    const bool reverse = (B != K || bi != ki) && allowed(f, ket.pair, bra.pair, screening);
    if (!forward && !reverse) continue;
    double bp[md_j_hermite_count(B)]{}, kp[md_j_hermite_count(K)]{};
    md_engine_contract<B, K>(bra, ket, f.md, forward, reverse, bp, kp);
    for (unsigned c = 0; c < md_j_hermite_count(B); ++c) {
      expected[bra.hermite_offset + c] += bp[c];
      magnitude[bra.hermite_offset + c] += std::abs(bp[c]);
    }
    for (unsigned c = 0; c < md_j_hermite_count(K); ++c) {
      expected[ket.hermite_offset + c] += kp[c];
      magnitude[ket.hermite_offset + c] += std::abs(kp[c]);
    }
    ++expected_counts.uniform_roots[5 * B + K];
    expected_counts.uniform_directions[5 * B + K] += forward;
    expected_counts.uniform_directions[5 * K + B] += reverse;
    expected_counts.uniform_summands[5 * B + K] +=
        forward * md_j_hermite_count(B) * md_j_hermite_count(K);
    expected_counts.uniform_summands[5 * K + B] +=
        reverse * md_j_hermite_count(B) * md_j_hermite_count(K);
  }
  f.poison_excluded_density(begin, end);
  if constexpr (!Count) f.md.work_counts = nullptr;
  block.run([&] { md_j_engine_potential<B, K, Count>(f.md, inventory, begin, end, screening); });
  f.restore_density_access();
  for (unsigned i = 0; i < expected.size(); ++i) {
    const double actual = f.potential[i];
    if (!std::isfinite(actual) || std::abs(actual - expected[i]) > 2e-13 * magnitude[i] + 1e-14) {
      std::fprintf(stderr, "B=%u K=%u nb=%u nk=%u mode=%u offset=%u got=%.17g expected=%.17g\n",
                   B, K, nb, nk, mode, i, actual, expected[i]);
      std::abort();
    }
  }
  if constexpr (Count) assert(std::memcmp(f.census.data(), &expected_counts, sizeof(expected_counts)) == 0);
  else {
    const MdJWorkCounts empty{};
    assert(std::memcmp(f.census.data(), &empty, sizeof(empty)) == 0);
  }
  assert(f.cursor == inventory.tasks + 1);
  assert(std::equal(f.original_density.begin(), f.original_density.end(), f.density.data()));
  f.check();
  ++completed;
}

/** Exercise the actual primitive hierarchy across tails, windows and replay.
 * Independent long-double Cauchy products identify potentially significant
 * leaves. Neither a prefix nor a coarse node may discard any such direction. */
void primitive_hierarchy(Block& block) {
  Fixture f({33, 0, 0, 0, 0}, 4);
  f.pack(block);
  std::vector<double> q(f.primitives.size());
  MdJPrimitiveScreenState state{};
  f.md.primitive_bounds=q.data(); f.md.primitive_screen=&state;
  f.md.maximum_pair_primitives=2;
  for (unsigned pair=0;pair<f.pairs.size();++pair) {
    f.pairs[pair].primitive_begin=0;
    f.pairs[pair].primitive_end=pair+1==f.pairs.size()?1:2;
  }
  const auto layout=f.md.hierarchy_layout.buckets[0];
  auto* nodes=f.md.hierarchy;
  for (double cutoff:{0.,1e-12,0.02}) for (unsigned mode:{0U,1U,2U,0U,3U,4U})
  for (auto window:{std::pair{0U,3U},std::pair{1U,2U},std::pair{1U,1U},std::pair{8U,9U}}) {
    const auto [begin,end]=window;
    state={}; f.md.primitive_maximum_bound=0.;
    std::vector<double> sources(q.size());
    for (unsigned position=0;position<q.size();++position) {
      const auto id=f.ordered[position], pair=f.primitives[id].pair;
      q[id]=position%7==0?0.:0.125*(1+position%11);
      if (mode==3 && position==31) q[id]=INFINITY;
      if (mode==4 && position>=8) q[id]=1e-15;
      f.md.primitive_maximum_bound=std::max(f.md.primitive_maximum_bound,q[id]);
      const auto system=unsigned(f.pairs[pair].system);
      sources[position]=mode==1 || system<begin || system>=end?0.:
          mode==2?(position==32?2e-14:0.):(position%4)*1e-14;
      if (mode==4) sources[position]=position<8 && system>=begin && system<end?5e-14:0.;
      if (mode==3 && position==30 && system>=begin && system<end) sources[position]=INFINITY;
      nodes[position].qd=sources[position];
      state.maximum_source_norm=std::max(state.maximum_source_norm,sources[position]);
      md_j_engine_count_source_norm(&state,sources[position]);
    }
    block.run([&] {md_j_engine_source_tails(&state);});
    block.run([&] {md_j_engine_hierarchy(f.md,0,begin,end,cutoff);});
    for (unsigned level=0;level<6;++level) {
      const auto width=1U<<level;
      for (unsigned group=0;group<layout.aligned/width;++group) {
        const auto node=nodes[layout.offsets[level]+group];
        double minimum=INFINITY, maximum=0.;
        for (unsigned j=group*width;j<(group+1)*width;++j) {
          minimum=std::min(minimum,nodes[j].minimum);
          maximum=std::max(maximum,nodes[j].qd);
        }
        assert(node.minimum==minimum && node.qd==maximum);
      }
    }
    for (unsigned j=q.size();j<layout.aligned;++j)
      assert(nodes[j].minimum==INFINITY && nodes[j].qd==0.);
    auto significant=[&](unsigned output,unsigned input) {
      const auto id=f.ordered[output], pair=f.primitives[id].pair;
      const auto source_pair=f.primitives[f.ordered[input]].pair;
      const auto system=unsigned(f.pairs[pair].system);
      if (system<begin || system>=end || f.pairs[source_pair].system!=int(system) || sources[input]==0.)
        return false;
      if (!std::isfinite(q[id]) || !std::isfinite(sources[input])) return true;
      return sources[input]>=nodes[output].minimum;
    };
    const auto schedule=md_j_tile_schedule(0,0);
    const auto inventory=md_j_tile_inventory(q.size(),q.size(),schedule.bra_width,schedule.ket_width,true);
    bool saw_significant=false;
    for (unsigned long long index=0;index<inventory.tasks;++index) {
      MdJTileTask task{}; assert(md_j_decode_tile(inventory,index,task));
      const bool admitted=md_j_engine_tile_admitted(f.md,0,0,schedule,task,cutoff);
      for (unsigned i=task.bra_begin;i<task.bra_begin+task.bra_count;++i)
      for (unsigned j=task.ket_begin;j<task.ket_begin+task.ket_count;++j)
        if (significant(i,j) || significant(j,i)) {
          saw_significant=true;
          assert(admitted);
          assert(state.active[0]>i && state.active[0]>j);
        }
    }
    if (mode==0 && begin==0 && end==3) assert(saw_significant);
    if (mode==1 || begin==end || begin==8) assert(state.active[0]==0);
    const auto frozen_active=state.active[0];
    // Independently sum all discarded Cauchy products, including BOTH the
    // prefix domain and leaf screening. Each primitive's share times
    // its parent's multiplicity must fit the complete public-AO error cap.
    for (unsigned output=0;output<q.size();++output) {
      const auto id=f.ordered[output], pair=f.primitives[id].pair;
      const auto system=unsigned(f.pairs[pair].system);
      if (system<begin || system>=end) continue;
      long double omitted=0.;
      for (unsigned input=0;input<q.size();++input) {
        const auto source_pair=f.primitives[f.ordered[input]].pair;
        if (sources[input]==0. || f.pairs[source_pair].system!=int(system)) continue;
        const bool kept=output<frozen_active && input<frozen_active &&
            sources[input]>=nodes[output].minimum;
        if (!kept) omitted+=static_cast<long double>(q[id])*sources[input];
      }
      assert(omitted*f.pairs[pair].primitive_end<=std::min(cutoff,1e-12));
    }
    for (unsigned long long index=0;index<inventory.tasks;++index) {
      MdJTileTask task{}; assert(md_j_decode_tile(inventory,index,task));
      const bool admitted=md_j_engine_tile_admitted(f.md,0,0,schedule,task,cutoff);
      for (unsigned i=task.bra_begin;i<task.bra_begin+task.bra_count;++i)
      for (unsigned j=task.ket_begin;j<task.ket_begin+task.ket_count;++j)
        if (i<frozen_active && j<frozen_active &&
            ((sources[j]!=0. && sources[j]>=nodes[i].minimum) ||
             (sources[i]!=0. && sources[i]>=nodes[j].minimum))) assert(admitted);
    }
    f.check();
  }
  ++completed;
}

void layouts(Block& block) {
  for (unsigned n : {0U, 1U, 15U, 16U, 17U, 31U, 32U, 33U, 127U, 129U}) {
    Fixture f({n, n ? n + 1 : 0, n / 2, n, n ? 3U : 0U}, 4);
    f.pack(block);
    f.build(block, 0, 3);
    f.build(block, 1, 2);
    f.build(block, 1, 1);
    f.build(block, 8, 9);
    ++completed;
  }
}

template <unsigned B, unsigned K> void angular_case(Block& block, unsigned section) {
  if (section == 1) evaluate<B, K>(block, 1, 1);
  if (section == 2) {
    constexpr unsigned sizes[]{1, 15, 16, 17, 31, 32, 33};
    constexpr auto n = sizes[(5 * B + K) % 7];
    evaluate<B, K>(block, n, n + (B != K));
  }
}
template <unsigned B, unsigned... K>
void row(Block& block, unsigned section, std::integer_sequence<unsigned, K...>) {
  (angular_case<B, B + K>(block, section), ...);
}
void all_pairs(Block& block, unsigned section) {
  row<0>(block, section, std::make_integer_sequence<unsigned, 5>{});
  row<1>(block, section, std::make_integer_sequence<unsigned, 4>{});
  row<2>(block, section, std::make_integer_sequence<unsigned, 3>{});
  row<3>(block, section, std::make_integer_sequence<unsigned, 2>{});
  row<4>(block, section, std::make_integer_sequence<unsigned, 1>{});
}
/** Independent long-double sums check the complete omitted source set. Bins
 * span subnormals, exponent transitions, tiny/huge allowances and overflow. */
void source_tail_budget(Block& block) {
  for (unsigned mode=0;mode<4;++mode) {
    MdJPrimitiveScreenState state{};
    std::vector<double> values{0.,-0.,INFINITY};
    for (int exponent=-1074;exponent<=1023;exponent+=7) {
      const double power=std::ldexp(1.0,exponent);
      if (mode==1 && exponent>-32) continue;
      if (mode==2 && exponent<32) continue;
      if (mode==3) continue;
      // Include internal mantissa boundaries and both adjacent FP64 values,
      // as well as normal/subnormal exponent transitions.
      for (double factor:{1.,1.25,1.5,1.75}) {
        const double boundary=power*factor;
        values.push_back(boundary);
        values.push_back(std::nextafter(boundary,0.));
        values.push_back(std::nextafter(boundary,INFINITY));
      }
    }
    for (double value:values) md_j_engine_count_source_norm(&state,value);
    block.run([&] {md_j_engine_source_tails(&state);});
    for (unsigned bin=0;bin<kMdJSourceNormBins;++bin) {
      const double threshold=__longlong_as_double(static_cast<long long>(bin)<<kMdJSourceNormShift);
      long double actual=0.;
      for (double value:values) if (value>0. && value<threshold) actual+=value;
      assert(state.source_norm_tails[bin]>=actual);
      if (bin) assert(state.source_norm_tails[bin]>=state.source_norm_tails[bin-1]);
    }
    for (double target:{0.,1e-200,0.75,1e200,double(INFINITY)})
    for (double budget:{0.,0x1p-1074,1e-300,1e-12,1.,1e300}) {
      const double threshold=md_j_engine_tail_threshold(&state,target,budget);
      assert(std::isfinite(threshold) && threshold>=0.);
      if (target==0. || !std::isfinite(target) || budget==0.) {
        assert(threshold==0.); continue;
      }
      long double actual=0.;
      for (double value:values) if (value>0. && value<threshold) actual+=value;
      assert(actual*target<=static_cast<long double>(budget));
    }
  }
  // A large census dominated by negligible sources must not charge every
  // omitted source the threshold value. Check useful gain without prescribing
  // an exact threshold or kernel/loop count.
  MdJPrimitiveScreenState state{};
  for (unsigned i=0;i<10000;++i) md_j_engine_count_source_norm(&state,1e-25);
  md_j_engine_count_source_norm(&state,1.0);
  block.run([&] {md_j_engine_source_tails(&state);});
  const double threshold=md_j_engine_tail_threshold(&state,1.0,1e-12);
  assert(threshold>1e-12/10001. && threshold<=1.0);
  ++completed;
}

int main(int argc, char** argv) {
  assert(argc == 2);
  Block block;
  switch (std::atoi(argv[1])) {
    case 0: layouts(block); break;
    case 1: case 2: all_pairs(block, std::atoi(argv[1])); break;
    case 3:
      evaluate<0, 4>(block, 17, 33, 1);
      evaluate<1, 3>(block, 15, 17, 2);
      evaluate<4, 4>(block, 17, 17, 2);
      evaluate<2, 4>(block, 17, 16, 3);
      evaluate<2, 3>(block, 16, 17, 4, 1, 2);
      evaluate<3, 3>(block, 33, 33, 4, 1, 1);
      evaluate<0, 2>(block, 0, 17);
      evaluate<2, 4>(block, 17, 0);
      evaluate<2, 2>(block, 0, 0);
      break;
    case 4:
      evaluate<0, 0>(block, 65, 65, 5);
      evaluate<4, 4>(block, 33, 33, 5);
      evaluate<1, 4, false>(block, 17, 17, 1);
      evaluate<4, 4, false>(block, 17, 17, 0, 0, 1, 0.0);
      evaluate<0, 4>(block, 33, 31, 6);
      evaluate<4, 4>(block, 33, 33, 6);
      evaluate<2, 3>(block, 1, 17, 6);
      break;
    case 5: screening_boundaries(); break;
    case 6: primitive_hierarchy(block); break;
    case 7: source_tail_budget(block); break;
    default: assert(false);
  }
  std::printf("PASS actual MD-J kernels, 128 host threads, %u cases, %s; not GPU qualification\n",
              completed, MD_J_HOST_ASAN ? "ASan+UBSan" : "UBSan");
}
"""
