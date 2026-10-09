"""Execute emitted K queue control flow with real concurrent host lane groups."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.integral.capabilities import CAPABILITY_MIXED_FOCK
from generativeqc_compiler.integral.cuda_schedule import ScheduleKind
from generativeqc_compiler.integral.production_emission import _streaming_fock_source
from generativeqc_compiler.integral.production_profile import resolve_production_profile
from generativeqc_compiler.integral.production_rys_tasks import (
    direct_rys_task_candidates,
)
from generativeqc_compiler.integral.production_selection import (
    supports_exchange_work_buckets,
)
from generativeqc_compiler.integral.shell_spec import shell_pair_class

if TYPE_CHECKING:
    from conftest import NativeCxx


@pytest.mark.parametrize(
    ("name", "task_parallel"),
    [("psss", False), ("dppp", False), ("dsds", False), ("dsss", True), ("dsds", True)],
)
@pytest.mark.parametrize("work_aware", [False, True])
def test_emitted_queue_executes_each_survivor_once(
    tmp_path: Path,
    name: str,
    task_parallel: bool,
    work_aware: bool,
    native_cxx: NativeCxx,
) -> None:
    """Cover sparse/dense/empty tails, canonical pairs and mixed-precision tags.

    Integral arithmetic is stubbed, but the actual emitted worker runs across
    every lane with blocking collectives. GPU matrix/sanitizer gates remain
    separate; this is an independent admission and synchronization census.
    """
    root = Path(__file__).resolve().parents[2]
    profile = resolve_production_profile(
        root / "python/generativeqc_compiler/integral/production_shell_classes.json",
        "sm_120",
    )
    selections = (
        direct_rys_task_candidates(profile) if task_parallel else profile.selections
    )
    selection = next(item for item in selections if item.spec.name == name)
    schedule = selection.fock_schedule or selection.schedule
    packed = schedule.kind == ScheduleKind.PACKED_TASKS
    mixed = selection.has_capability(CAPABILITY_MIXED_FOCK)
    class_name = name[0].upper() + name[1:]
    prefix = f"generated_{name}"
    source = _streaming_fock_source(selection)
    worker_name = f"{prefix}_{'work_' if work_aware else ''}streaming_fock"
    declaration = (
        f"__device__ __forceinline__ unsigned {prefix}_exchange_work_bucket("
        if work_aware
        else f"__device__ __forceinline__ void {prefix}_sort_exchange_queue("
    )
    start = source.index(declaration)
    marker = source.index(f"__device__ __forceinline__ void {worker_name}(")
    worker = source[start : source.index("\n}\n", marker) + 3]
    first_class = shell_pair_class(*selection.spec.angular[:2])
    second_class = shell_pair_class(*selection.spec.angular[2:])
    high_class, low_class = (
        max(first_class, second_class),
        min(first_class, second_class),
    )
    width = 32 if packed else schedule.tasks_per_block
    consumer = "packed_fock_lane" if packed else "subgroup_fock_task"
    mixed_consumer = "packed_mixed_fock_lane" if packed else "mixed_subgroup_fock_task"
    storage = (
        f"Generated{class_name}{'PackedFockLane' if packed else 'SubgroupFock'}Storage"
    )
    subgroup_parameters = "" if packed else ", unsigned lane, unsigned"
    only_leader = "true" if packed else "lane == 0U"
    first_slot = "threadIdx.x % 32U == 0U" if packed else "index == 0U"
    precision_parameters = (
        "unsigned state, unsigned long long* fp64, unsigned long long* fp32"
        if mixed
        else "unsigned long long* fp64"
    )
    counter_pointer = "state == 3U ? fp32 : fp64" if mixed else "fp64"
    precision_arguments = "true, .4, " if mixed else ""
    extra_counter = ", &fp32" if mixed else ""
    driver = f"""
#include <algorithm>
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>
#include "scf/generated_shell_task.hpp"
#define __device__
#define __forceinline__
#define __shared__ static
constexpr unsigned block_threads = {schedule.block_threads};
constexpr unsigned execution_width = {width};
int observed_buckets[block_threads];
std::vector<unsigned> bucket_labels;
std::atomic<bool> homogeneous_error = false;
struct Completion {{
  unsigned begin, count;
  void operator()() noexcept {{
    int first_bucket = -1;
    for (unsigned index = begin; index < begin + count; ++index) {{
      auto& observed = observed_buckets[index];
      if ({str(work_aware).lower()} && observed >= 0) {{
        if (first_bucket >= 0 && first_bucket != observed) homogeneous_error = true;
        first_bucket = observed;
      }}
      observed = -1;
    }}
  }}
}};
std::barrier rendezvous(block_threads, Completion{{0, block_threads}});
auto warp_rendezvous = [] {{
  std::vector<std::unique_ptr<std::barrier<Completion>>> groups;
  for (unsigned warp = 0; warp < block_threads / 32; ++warp)
    groups.push_back(std::make_unique<std::barrier<Completion>>(
        32, Completion{{32 * warp, 32}}));
  return groups;
}}();
thread_local struct Lane {{ unsigned x; }} threadIdx;
#define __syncthreads() rendezvous.arrive_and_wait()
void __syncwarp(unsigned) {{ warp_rendezvous[threadIdx.x / 32]->arrive_and_wait(); }}
bool ballot_votes[block_threads];
unsigned __ballot_sync(unsigned, bool keep) {{
  ballot_votes[threadIdx.x] = keep;
  __syncwarp(0xffffffffU);
  unsigned result = 0;
  for (unsigned lane = 0; lane < 32; ++lane)
    if (ballot_votes[threadIdx.x / 32 * 32 + lane]) result |= 1U << lane;
  __syncwarp(0xffffffffU);
  return result;
}}
unsigned __popc(unsigned value) {{ return __builtin_popcount(value); }}
template<class Integer> Integer atomicAdd(Integer* pointer, Integer value) {{
  return std::atomic_ref<Integer>(*pointer).fetch_add(value);
}}
using Topology = generativeqc::scf::detail::GeneratedShellPairStream;
using Consumer = generativeqc::scf::detail::GeneratedFockConsumer;
using Queue = generativeqc::scf::detail::GeneratedExchangeTaskSchedule;
using Generated{class_name}ShellTask = generativeqc::scf::detail::GeneratedShellTask;
using Generated{class_name}PrimitivePairData = generativeqc::scf::detail::GeneratedPrimitivePairData;
struct Generated{class_name}Vec3 {{ double x, y, z; }};
struct {storage} {{}};
struct Generated{class_name}MixedSubgroupFockStorage {{}};
constexpr unsigned kGenerated{class_name}FockBlockThreads = block_threads;
using Pair = std::tuple<unsigned, unsigned, bool>;
std::vector<Pair> actual;
std::mutex output_mutex;
std::atomic<unsigned> batches;
unsigned pattern;
bool accepted(const Topology& topology, unsigned bra, unsigned ket) {{
  if (!topology.active[topology.shell_pair_systems[bra]]) return false;
  if (pattern == 0) return false;
  if (pattern == 1) return ket % 31 == 0;
  if (pattern == 2) return true;
  return (bra * 13 + ket * 7) % 5 < 2;
}}
template<bool Unrestricted> bool {prefix}_stream_survives(
    const Topology& topology, unsigned bra, unsigned ket, double, double* bound) {{
  *bound = double(ket % 17 + 1) / 20;
  return accepted(topology, bra, ket);
}}
unsigned {prefix}_stream_coarse_ket_end(
    const Topology&, unsigned, unsigned begin, unsigned end, double, double) {{
  return pattern == 3 ? begin + (end - begin) / 2 : end;
}}
void {prefix}_record_fock_precision({precision_parameters}) {{
  auto* counter = {counter_pointer};
  if (counter) atomicAdd(counter, 1ULL);
}}
void {prefix}_stream_populate_task(
    const Topology&, unsigned bra, unsigned ket, Generated{class_name}ShellTask& task) {{
  task.shell_pair[0] = bra; task.shell_pair[1] = ket;
}}
template<bool Unrestricted> void {prefix}_{consumer}(
    const Generated{class_name}ShellTask* tasks, const Generated{class_name}PrimitivePairData*,
    const std::int64_t*, const double*, const Generated{class_name}Vec3*, double,
    const double*, const double*, double*, std::size_t index, {storage}&{subgroup_parameters}) {{
  if ({only_leader}) {{
    std::lock_guard lock(output_mutex);
    actual.emplace_back(tasks[index].shell_pair[0], tasks[index].shell_pair[1], false);
    observed_buckets[threadIdx.x] = bucket_labels[tasks[index].shell_pair[1]];
    if ({first_slot}) ++batches;
  }}
}}
"""
    if mixed:
        driver += f"""
template<bool Unrestricted> void {prefix}_{mixed_consumer}(
    const Generated{class_name}ShellTask* tasks, const Generated{class_name}PrimitivePairData*,
    const std::int64_t*, const double*, const Generated{class_name}Vec3*, double,
    const double*, const double*, double*, std::size_t index,
    Generated{class_name}MixedSubgroupFockStorage&{subgroup_parameters}) {{
  if ({only_leader}) {{
    std::lock_guard lock(output_mutex);
    actual.emplace_back(tasks[index].shell_pair[0], tasks[index].shell_pair[1], true);
    observed_buckets[threadIdx.x] = bucket_labels[tasks[index].shell_pair[1]];
    if ({first_slot}) ++batches;
  }}
}}
"""
    driver += (
        worker
        + f"""
int main() {{
  for (auto consumer : {{{"Consumer::Exchange, Consumer::HartreeFockExchange" if work_aware else "Consumer::HartreeFock, Consumer::Coulomb, Consumer::Exchange, Consumer::HartreeFockExchange"}}}) {{
  for (unsigned mode = {3 if work_aware else 0}; mode < {4 if work_aware else 3}; ++mode) {{
    for (pattern = 0; pattern < 4; ++pattern) {{
      for (unsigned count : {{0U, 1U, execution_width - 1U, execution_width,
                             execution_width + 1U, 3U * execution_width + 5U}}) {{
        std::vector<std::int32_t> systems;
        std::vector<std::uint32_t> order;
        std::uint32_t offsets[30]{{}};
        if ({str(high_class == low_class).lower()}) {{
          systems.resize(count + 7, 0);
          std::fill(systems.begin() + count, systems.end(), 1);
          order.resize(systems.size());
          std::iota(order.begin(), order.end(), 0);
          offsets[{high_class} * 3] = 0;
          offsets[{high_class} * 3 + 1] = count;
          offsets[{high_class} * 3 + 2] = order.size();
        }} else {{
          systems = {{0, 1}};
          order = {{0, 1}};
          offsets[{high_class} * 3] = 0;
          offsets[{high_class} * 3 + 1] = 1;
          offsets[{high_class} * 3 + 2] = 2;
          offsets[{low_class} * 3] = order.size();
          for (unsigned index = 0; index < count; ++index) {{
            order.push_back(systems.size()); systems.push_back(0);
          }}
          offsets[{low_class} * 3 + 1] = order.size();
          for (unsigned index = 0; index < 7; ++index) {{
            order.push_back(systems.size()); systems.push_back(1);
          }}
          offsets[{low_class} * 3 + 2] = order.size();
        }}
        std::vector<std::int64_t> primitive_offsets(systems.size() + 1, 0);
        std::vector<std::int32_t> first_shells(systems.size()), second_shells(systems.size());
        std::vector<std::int64_t> shell_offsets(2 * systems.size() + 1, 0);
        std::vector<unsigned> buckets(systems.size());
        constexpr unsigned lengths[] = {{1, 2, 3, 4, 6, 16, 128}};
        for (unsigned index = 0; index < systems.size(); ++index) {{
          const unsigned first_length = lengths[index % 7];
          const unsigned second_length = lengths[(index / 7) % 7];
          first_shells[index] = 2 * index;
          second_shells[index] = 2 * index + 1;
          shell_offsets[2 * index + 1] = shell_offsets[2 * index] + first_length;
          shell_offsets[2 * index + 2] = shell_offsets[2 * index + 1] + second_length;
          const unsigned products = first_length * second_length;
          primitive_offsets[index + 1] = primitive_offsets[index] + products;
          unsigned magnitude = 0;
          for (unsigned ceiling = 2; ceiling <= products && magnitude < 3; ceiling *= 2)
            ++magnitude;
          buckets[index] = 2 * magnitude + (first_length > 1 && second_length > 1);
        }}
        bucket_labels = buckets;
        std::uint8_t active[]{{1, 0}};
        Topology topology{{}};
        topology.batch_size = 2;
        topology.shell_pair_systems = systems.data();
        topology.pair_order = order.data(); topology.pair_class_offsets = offsets;
        topology.active = active; topology.fock_consumer = consumer;
        topology.exchange_task_schedule = static_cast<Queue>(mode);
        topology.shell_pair_first = first_shells.data();
        topology.shell_pair_second = second_shells.data();
        topology.shell_primitive_offsets = shell_offsets.data();
        std::vector<Pair> expected;
        unsigned expected_batches = 0;
        unsigned long long expected_fp64 = 0, expected_fp32 = 0;
        for (unsigned ordinal = offsets[{high_class} * 3];
             ordinal < offsets[{high_class} * 3 + 2]; ++ordinal) {{
          unsigned bra = order[ordinal], system = systems[bra], survivors = 0;
          unsigned bucket_counts[32]{{}};
          unsigned begin = offsets[{low_class} * 3 + system];
          unsigned end = {prefix}_stream_coarse_ket_end(
              topology, bra, begin, offsets[{low_class} * 3 + system + 1], 1, 1);
          for (unsigned ket_ordinal = begin; ket_ordinal < end; ++ket_ordinal) {{
            unsigned ket = order[ket_ordinal];
            if ({str(high_class == low_class).lower()} && bra < ket) continue;
            if (!accepted(topology, bra, ket)) continue;
            bool fp32 = {str(mixed).lower()} && double(ket % 17 + 1) / 20 < .4;
            expected.emplace_back(bra, ket, fp32); ++survivors;
            ++bucket_counts[buckets[ket]];
            if (fp32) ++expected_fp32;
            else ++expected_fp64;
          }}
          if ({str(work_aware).lower()}) {{
            for (auto bucket_count : bucket_counts)
              expected_batches += (bucket_count + execution_width - 1) / execution_width;
          }} else expected_batches += (survivors + execution_width - 1) / execution_width;
        }}
        actual.clear(); batches = 0;
        unsigned head = 0;
        unsigned long long fp64 = 0, fp32 = 0;
        std::vector<std::thread> lanes;
        for (unsigned lane = 0; lane < block_threads; ++lane)
          lanes.emplace_back([&, lane] {{
            threadIdx.x = lane;
            {worker_name}<false>(&topology, nullptr, primitive_offsets.data(),
                nullptr, nullptr, 1., {precision_arguments}nullptr, nullptr, nullptr,
                &head, &fp64{extra_counter});
          }});
        for (auto& lane : lanes) lane.join();
        std::sort(expected.begin(), expected.end()); std::sort(actual.begin(), actual.end());
        if (homogeneous_error || actual != expected || fp64 != expected_fp64 || fp32 != expected_fp32 ||
            (mode != 0 && (consumer == Consumer::Exchange ||
                          consumer == Consumer::HartreeFockExchange) &&
             batches != expected_batches)) {{
          std::cerr << "queue census failed: mode=" << mode << " pattern=" << pattern
                    << " count=" << count << " actual=" << actual.size()
                    << " expected=" << expected.size() << '\\n';
          return 1;
        }}
      }}
    }}
  }}
  }}
}}
"""
    )
    driver_path = tmp_path / "queue.cpp"
    driver_path.write_text(driver)
    executable = tmp_path / "queue"
    native_cxx.build_executable(
        (driver_path,),
        executable,
        compile_args=("-std=c++20", "-O1", "-pthread", f"-I{root / 'src'}"),
        link_args=("-pthread",),
        compile_timeout=60,
    )
    subprocess.run([str(executable)], check=True, timeout=120)


def test_prepared_schedule_parser_is_explicit_and_fail_closed(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Default work and explicit rollback freeze before later environment edits."""
    root = Path(__file__).resolve().parents[2]
    registry = tmp_path / "scf/aot_shell_registry.hpp"
    registry.parent.mkdir()
    registry.write_text(
        "#include <cstdint>\n"
        "namespace generativeqc::scf::generated {\n"
        "inline std::uint64_t enabled_rys_fock_shell_class_mask() { return 2; }\n"
        "inline std::uint64_t enabled_rys_task_fock_shell_class_mask() { return 24; }\n"
        "inline std::uint64_t preferred_task_mask = 8;\n"
        "inline std::uint64_t preferred_rys_task_fock_shell_class_mask() { return preferred_task_mask; }\n"
        "inline std::uint64_t enabled_k_block_fock_shell_class_mask() { return 4; }\n"
        "inline void launch_shell_class_rys_streaming_fock() {}\n"
        "inline void launch_shell_class_rys_task_streaming_fock() {}\n"
        "inline void launch_shell_class_rys_task_work_streaming_fock() {}\n"
        "inline void launch_shell_class_k_block_streaming_fock() {}\n"
        "inline void launch_shell_class_streaming_fock() {}\n"
        "inline void launch_shell_class_work_streaming_fock() {}\n}\n"
    )
    driver = tmp_path / "selection.cpp"
    driver.write_text(
        r"""
#include <cassert>
#include "scf/cuda/direct_fock_lowering.hpp"
int main() {
  using Schedule = generativeqc::scf::detail::GeneratedExchangeTaskSchedule;
  using generativeqc::scf::cuda_execution::prepare_direct_exchange_task_schedule;
  constexpr const char* variable = "GENERATIVEQC_DIRECT_K_TASK_SCHEDULE";
  const generativeqc::scf::detail::GeneratedShellPairStream fallback_topology{};
  assert(fallback_topology.exchange_task_schedule == Schedule::Incumbent);
  unsetenv(variable);
  const auto frozen = prepare_direct_exchange_task_schedule();
  assert(frozen == Schedule::Work);
  for (const char* value : {"", "work"}) {
    setenv(variable, value, 1);
    assert(prepare_direct_exchange_task_schedule() == Schedule::Work);
  }
  setenv(variable, "fill", 1);
  assert(prepare_direct_exchange_task_schedule() == Schedule::Fill);
  assert(frozen == Schedule::Work);
  setenv(variable, "incumbent", 1);
  assert(prepare_direct_exchange_task_schedule() == Schedule::Incumbent);
  assert(frozen == Schedule::Work);
  setenv(variable, "primitive", 1);
  assert(frozen == Schedule::Work);
  assert(prepare_direct_exchange_task_schedule() == Schedule::Primitive);
  setenv(variable, "work", 1);
  assert(frozen == Schedule::Work);
  assert(prepare_direct_exchange_task_schedule() == Schedule::Work);
  setenv(variable, "typo", 1);
  try {
    (void)prepare_direct_exchange_task_schedule();
    return 1;
  } catch (const std::invalid_argument&) {}
  using namespace generativeqc::scf::cuda_execution;
  constexpr const char* lowering = "GENERATIVEQC_DIRECT_K_FOCK_LOWERING";
  unsetenv(lowering);
  const auto default_mask = prepare_direct_fock_rys_task_mask();
  assert(default_mask == 8);
  assert(prepare_direct_fock_rys_mask(true) == 0);
  assert(prepare_direct_fock_rys_mask(false) == 0);
  assert(prepare_direct_fock_k_block_mask() == 0);
  setenv(lowering, "", 1);
  assert(prepare_direct_fock_rys_task_mask() == default_mask);
  generativeqc::scf::generated::preferred_task_mask = 0;
  assert(prepare_direct_fock_rys_task_mask() == 0);
  assert(default_mask == 8);
  for (const char* mode : {"rys", "block"}) {
    setenv(lowering, mode, 1);
    assert(prepare_direct_fock_rys_task_mask() == 0);
    assert(prepare_direct_fock_rys_mask(true) == (std::strcmp(mode, "rys") == 0 ? 2 : 0));
    assert(prepare_direct_fock_k_block_mask() == (std::strcmp(mode, "block") == 0 ? 4 : 0));
  }
  setenv(lowering, "rys-task", 1);
  const auto task_mask = prepare_direct_fock_rys_task_mask();
  assert(task_mask == 24);
  assert(prepare_direct_fock_rys_mask(true) == 0);
  assert(prepare_direct_fock_k_block_mask() == 0);
  const DirectExchangeSelection task_selection{0, 0, task_mask};
  assert(direct_fock_streaming_launcher(task_selection, 4, false) ==
      generativeqc::scf::generated::launch_shell_class_rys_task_streaming_fock);
  assert(direct_fock_streaming_launcher(task_selection, 0, false) ==
      generativeqc::scf::generated::launch_shell_class_streaming_fock);
  setenv(lowering, "incumbent", 1);
  assert(prepare_direct_fock_rys_task_mask() == 0);
  assert(task_mask == 24);

  // Exhaust both independent controls, reachable coverage and spin contracts.
  namespace registry = generativeqc::scf::generated;
  registry::preferred_task_mask = 8;
  constexpr const char* j_lowering = "GENERATIVEQC_DIRECT_J_FOCK_LOWERING";
  setenv(j_lowering, "typo", 1);
  const char* modes[] = {nullptr, "", "incumbent", "rys", "block", "rys-task"};
  const char* schedules[] = {nullptr, "", "incumbent", "fill", "primitive", "work"};
  for (const char* mode : modes) {
    for (const char* schedule : schedules) {
      for (const std::uint64_t coverage : {0U, 2U, 4U, 8U, 16U, 31U}) {
        if (mode == nullptr) unsetenv(lowering); else setenv(lowering, mode, 1);
        if (schedule == nullptr) unsetenv(variable); else setenv(variable, schedule, 1);
        const auto selection = prepare_direct_exchange_selection(coverage);
        const bool default_lowering = mode == nullptr || *mode == '\0';
        const bool work = schedule == nullptr || *schedule == '\0' ||
                          std::strcmp(schedule, "work") == 0;
        const std::uint64_t rys = !default_lowering && std::strcmp(mode, "rys") == 0 ? 2 : 0;
        const std::uint64_t block = !default_lowering && std::strcmp(mode, "block") == 0 ? 4 : 0;
        const std::uint64_t task = default_lowering ? 8 :
            (std::strcmp(mode, "rys-task") == 0 ? 24 : 0);
        const auto expected_schedule = work ? Schedule::Work :
            (std::strcmp(schedule, "fill") == 0 ? Schedule::Fill :
             (std::strcmp(schedule, "primitive") == 0 ? Schedule::Primitive : Schedule::Incumbent));
        assert(selection.rys_fock_mask == (rys & coverage));
        assert(selection.k_block_fock_mask == (block & coverage));
        assert(selection.rys_task_fock_mask == (task & coverage));
        assert(selection.task_schedule == expected_schedule);
        setenv(lowering, "typo", 1);
        setenv(variable, "typo", 1);
        registry::preferred_task_mask = 0;
        for (bool unrestricted : {false, true}) {
          for (unsigned cls = 0; cls < 6; ++cls) {
            const auto bit = std::uint64_t{1} << cls;
            const auto expected = (task & coverage & bit) ?
                (work ? registry::launch_shell_class_rys_task_work_streaming_fock :
                        registry::launch_shell_class_rys_task_streaming_fock) :
                (!unrestricted && (block & coverage & bit)) ? registry::launch_shell_class_k_block_streaming_fock :
                (rys & coverage & bit) ? registry::launch_shell_class_rys_streaming_fock :
                work ? registry::launch_shell_class_work_streaming_fock : registry::launch_shell_class_streaming_fock;
            assert(direct_fock_streaming_launcher(selection, cls, unrestricted) == expected);
          }
        }
        registry::preferred_task_mask = 8;
      }
    }
  }
  for (const char* bad : {"typo", "work", "fill"}) {
    setenv(lowering, bad, 1);
    unsetenv(variable);
    try {
      (void)prepare_direct_exchange_selection(31);
      return 1;
    } catch (const std::invalid_argument&) {}
  }
  unsetenv(lowering);
  setenv(variable, "typo", 1);
  try {
    (void)prepare_direct_exchange_selection(31);
    return 1;
  } catch (const std::invalid_argument&) {}
  for (const char* bad : {"block", "rys-task", "typo"}) {
    setenv(j_lowering, bad, 1);
    try {
      (void)prepare_direct_fock_rys_mask(false);
      return 1;
    } catch (const std::invalid_argument&) {}
  }
  for (const char* mode : {"", "incumbent", "rys"}) {
    setenv(j_lowering, mode, 1);
    const auto mask = prepare_direct_fock_rys_mask(false);
    assert(mask == (std::strcmp(mode, "rys") == 0 ? 2 : 0));
    assert(direct_fock_streaming_launcher(mask, 3) == registry::launch_shell_class_streaming_fock);
  }
  // Defensive precedence remains deterministic even for overlapping supplied masks.
  DirectExchangeSelection overlap{2, 2, 2, Schedule::Work};
  assert(direct_fock_streaming_launcher(overlap, 1, true) == registry::launch_shell_class_rys_task_work_streaming_fock);
  overlap.rys_task_fock_mask = 0;
  assert(direct_fock_streaming_launcher(overlap, 1, false) == registry::launch_shell_class_k_block_streaming_fock);
  assert(direct_fock_streaming_launcher(overlap, 1, true) == registry::launch_shell_class_rys_streaming_fock);
}
"""
    )
    executable = tmp_path / "selection"
    native_cxx.build_executable(
        (driver,),
        executable,
        compile_args=("-std=c++20", f"-I{tmp_path}", f"-I{root / 'src'}"),
        compile_timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)


def test_work_specialization_is_separate_from_alternative_lowerings() -> None:
    """Whole-CTA classes and optional lowering variants keep their own workers."""
    root = Path(__file__).resolve().parents[2]
    profile = resolve_production_profile(
        root / "python/generativeqc_compiler/integral/production_shell_classes.json",
        "sm_120",
    )
    for name in ("ddds", "dpps", "ppps", "psss", "ddpp"):
        selection = next(item for item in profile.selections if item.spec.name == name)
        source = _streaming_fock_source(selection)
        incumbent = _streaming_fock_source(selection, include_work_buckets=False)
        assert "_work_streaming_kernel" not in incumbent
        assert ("_work_streaming_kernel" in source) == supports_exchange_work_buckets(
            selection
        )
        assert source.startswith(incumbent)
