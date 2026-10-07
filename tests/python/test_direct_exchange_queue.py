"""Execute emitted K queue control flow with real concurrent host lane groups."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from generativeqc_compiler.integral.capabilities import CAPABILITY_MIXED_FOCK
from generativeqc_compiler.integral.cuda_schedule import ScheduleKind
from generativeqc_compiler.integral.production_emission import _streaming_fock_source
from generativeqc_compiler.integral.production_profile import resolve_production_profile
from generativeqc_compiler.integral.shell_spec import shell_pair_class


@pytest.mark.parametrize("name", ["psss", "dppp", "dsds"])
def test_emitted_queue_executes_each_survivor_once(tmp_path: Path, name: str) -> None:
    """Cover sparse/dense/empty tails, canonical pairs and mixed-precision tags.

    Integral arithmetic is stubbed, but the actual emitted worker runs across
    every lane with blocking collectives. GPU matrix/sanitizer gates remain
    separate; this is an independent admission and synchronization census.
    """
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++20 host compiler")
    root = Path(__file__).resolve().parents[2]
    profile = resolve_production_profile(
        root / "python/generativeqc_compiler/integral/production_shell_classes.json",
        "sm_120",
    )
    selection = next(item for item in profile.selections if item.spec.name == name)
    schedule = selection.fock_schedule or selection.schedule
    packed = schedule.kind == ScheduleKind.PACKED_TASKS
    mixed = selection.has_capability(CAPABILITY_MIXED_FOCK)
    class_name = name[0].upper() + name[1:]
    prefix = f"generated_{name}"
    source = _streaming_fock_source(selection)
    start = source.index(
        f"__device__ __forceinline__ void {prefix}_sort_exchange_queue("
    )
    marker = source.index(f"__device__ __forceinline__ void {prefix}_streaming_fock(")
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
    first_slot = "threadIdx.x == 0U" if packed else "index == 0U"
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
std::barrier rendezvous(block_threads);
thread_local struct Lane {{ unsigned x; }} threadIdx;
#define __syncthreads() rendezvous.arrive_and_wait()
void __syncwarp(unsigned) {{ rendezvous.arrive_and_wait(); }}
bool ballot_votes[32];
unsigned __ballot_sync(unsigned, bool keep) {{
  ballot_votes[threadIdx.x] = keep;
  rendezvous.arrive_and_wait();
  unsigned result = 0;
  for (unsigned lane = 0; lane < 32; ++lane)
    if (ballot_votes[lane]) result |= 1U << lane;
  rendezvous.arrive_and_wait();
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
    if ({first_slot}) ++batches;
  }}
}}
"""
    driver += (
        worker
        + f"""
int main() {{
  for (auto consumer : {{Consumer::HartreeFock, Consumer::Coulomb,
                        Consumer::Exchange, Consumer::HartreeFockExchange}}) {{
  for (unsigned mode = 0; mode < 3; ++mode) {{
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
        for (unsigned index = 0; index < systems.size(); ++index)
          primitive_offsets[index + 1] = primitive_offsets[index] + (index * 7) % 19 + 1;
        std::uint8_t active[]{{1, 0}};
        Topology topology{{}};
        topology.batch_size = 2;
        topology.shell_pair_systems = systems.data();
        topology.pair_order = order.data(); topology.pair_class_offsets = offsets;
        topology.active = active; topology.fock_consumer = consumer;
        topology.exchange_task_schedule = static_cast<Queue>(mode);
        std::vector<Pair> expected;
        unsigned expected_batches = 0;
        unsigned long long expected_fp64 = 0, expected_fp32 = 0;
        for (unsigned ordinal = offsets[{high_class} * 3];
             ordinal < offsets[{high_class} * 3 + 2]; ++ordinal) {{
          unsigned bra = order[ordinal], system = systems[bra], survivors = 0;
          unsigned begin = offsets[{low_class} * 3 + system];
          unsigned end = {prefix}_stream_coarse_ket_end(
              topology, bra, begin, offsets[{low_class} * 3 + system + 1], 1, 1);
          for (unsigned ket_ordinal = begin; ket_ordinal < end; ++ket_ordinal) {{
            unsigned ket = order[ket_ordinal];
            if ({str(high_class == low_class).lower()} && bra < ket) continue;
            if (!accepted(topology, bra, ket)) continue;
            bool fp32 = {str(mixed).lower()} && double(ket % 17 + 1) / 20 < .4;
            expected.emplace_back(bra, ket, fp32); ++survivors;
            if (fp32) ++expected_fp32;
            else ++expected_fp64;
          }}
          expected_batches += (survivors + execution_width - 1) / execution_width;
        }}
        actual.clear(); batches = 0;
        unsigned head = 0;
        unsigned long long fp64 = 0, fp32 = 0;
        std::vector<std::thread> lanes;
        for (unsigned lane = 0; lane < block_threads; ++lane)
          lanes.emplace_back([&, lane] {{
            threadIdx.x = lane;
            {prefix}_streaming_fock<false>(&topology, nullptr, primitive_offsets.data(),
                nullptr, nullptr, 1., {precision_arguments}nullptr, nullptr, nullptr,
                &head, &fp64{extra_counter});
          }});
        for (auto& lane : lanes) lane.join();
        std::sort(expected.begin(), expected.end()); std::sort(actual.begin(), actual.end());
        if (actual != expected || fp64 != expected_fp64 || fp32 != expected_fp32 ||
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
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            "-pthread",
            "-I",
            str(root / "src"),
            str(driver_path),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=60,
    )
    subprocess.run([str(executable)], check=True, timeout=120)


def test_prepared_schedule_parser_is_explicit_and_fail_closed(tmp_path: Path) -> None:
    """Default fill and explicit rollback freeze before later environment edits."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    root = Path(__file__).resolve().parents[2]
    registry = tmp_path / "scf/aot_shell_registry.hpp"
    registry.parent.mkdir()
    registry.write_text(
        "#include <cstdint>\n"
        "namespace generativeqc::scf::generated {\n"
        "inline std::uint64_t enabled_rys_fock_shell_class_mask() { return 0; }\n"
        "inline std::uint64_t enabled_k_block_fock_shell_class_mask() { return 0; }\n"
        "inline void launch_shell_class_rys_streaming_fock() {}\n"
        "inline void launch_shell_class_k_block_streaming_fock() {}\n"
        "inline void launch_shell_class_streaming_fock() {}\n}\n"
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
  assert(frozen == Schedule::Fill);
  for (const char* value : {"", "fill"}) {
    setenv(variable, value, 1);
    assert(prepare_direct_exchange_task_schedule() == Schedule::Fill);
  }
  setenv(variable, "incumbent", 1);
  assert(prepare_direct_exchange_task_schedule() == Schedule::Incumbent);
  assert(frozen == Schedule::Fill);
  setenv(variable, "primitive", 1);
  assert(frozen == Schedule::Fill);
  assert(prepare_direct_exchange_task_schedule() == Schedule::Primitive);
  setenv(variable, "typo", 1);
  try {
    (void)prepare_direct_exchange_task_schedule();
    return 1;
  } catch (const std::invalid_argument&) {}
}
"""
    )
    executable = tmp_path / "selection"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-I",
            str(tmp_path),
            "-I",
            str(root / "src"),
            str(driver),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)
