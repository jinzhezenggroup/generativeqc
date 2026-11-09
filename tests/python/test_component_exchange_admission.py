"""Execute emitted admission contracts with host SIMT stand-ins, not ERI oracles.

Real CUDA raw-K, graph, sanitizer and complete-endpoint qualification remain
separate requirements. This bounded host probe catches domain loss, duplicate
publication, partial-warps and unsupported-consumer fallthrough in ordinary CI.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.integral.production_component_queue import (
    component_exchange_admission_loop,
)

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import NativeCxx


@pytest.mark.parametrize("different_pairs", (False, True))
def test_compiled_admission_domain(
    tmp_path: Path, required_native_cxx: NativeCxx, different_pairs: bool
) -> None:
    """Compare actual emitted admission with an independent Cartesian domain."""
    high_class, low_class = (3, 1) if different_pairs else (1, 1)
    queue = component_exchange_admission_loop(
        prefix="probe",
        high_pair_class=high_class,
        low_pair_class=low_class,
        system_density_bound="1.0",
        fock_task="probe_task",
        record_precision="++admitted;",
    )
    driver = tmp_path / "admission.cpp"
    driver.write_text(
        r"""
#include <algorithm>
#include <atomic>
#include <barrier>
#include <bit>
#include <cassert>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

namespace generativeqc::scf::detail {
enum class GeneratedFockConsumer { Exchange, Coulomb };
}
using Consumer = generativeqc::scf::detail::GeneratedFockConsumer;
struct Index { unsigned x; };
thread_local Index threadIdx;
std::barrier block_barrier(64), warp_barrier(32);
std::atomic<unsigned> ballot_bits;
unsigned __ballot_sync(unsigned mask, bool keep) {
  assert(mask == 0xffffffffU && threadIdx.x < 32);
  if (keep) ballot_bits.fetch_or(1U << threadIdx.x);
  warp_barrier.arrive_and_wait();
  const unsigned result = ballot_bits.load();
  warp_barrier.arrive_and_wait();
  if (threadIdx.x == 0) ballot_bits = 0;
  warp_barrier.arrive_and_wait();
  return result;
}
unsigned __popc(unsigned value) { return std::popcount(value); }
void __syncthreads() { block_barrier.arrive_and_wait(); }
unsigned atomicAdd(std::uint32_t* pointer, unsigned amount) {
  return std::atomic_ref(*pointer).fetch_add(amount);
}
#define __shared__ static
struct Topology {
  int batch_size = 7;
  Consumer fock_consumer = Consumer::Exchange;
  const void* shell_pair_density_bounds = this;
  const void* system_pair_density_bounds = this;
  std::vector<std::uint32_t> pair_class_offsets, pair_order;
  std::vector<int> shell_pair_systems;
  std::vector<double> shell_pair_bounds;
} topology;
struct Task { unsigned bra, ket; };
std::vector<std::pair<unsigned, unsigned>> actual;
unsigned admitted = 0;
std::atomic<unsigned> fallthrough;
template<bool Unrestricted>
bool probe_stream_survives(const Topology&, unsigned bra, unsigned ket,
                          double, double* bound) {
  *bound = 1.0;
  return (3 * bra + ket) % 17 == 0;
}
void probe_stream_populate_task(const Topology&, unsigned bra, unsigned ket,
                                Task& task) { task = {bra, ket}; }
void probe_task(Task* task, const void*, const void*, const void*, const void*,
                double, const void*, const void*, const void*, unsigned) {
  if (threadIdx.x == 0) actual.emplace_back(task->bra, task->ket);
}
unsigned probe_stream_coarse_ket_end(const Topology& input, unsigned bra,
                                    unsigned begin, unsigned end,
                                    double density, double tolerance) {
  while (end > begin && input.shell_pair_bounds[bra] *
          input.shell_pair_bounds[input.pair_order[end-1]] * density < tolerance)
    --end;
  return end;
}
template<bool Unrestricted>
void worker(std::uint32_t* head, double screening_tolerance) {
  const auto stride = static_cast<std::size_t>(topology.batch_size) + 1;
  const auto bra_begin = topology.pair_class_offsets[HIGH_CLASS * stride];
  const auto bra_end = topology.pair_class_offsets[HIGH_CLASS * stride + topology.batch_size];
  __shared__ std::uint32_t candidate_ordinal, bra_ordinal;
  __shared__ Task stream_task[1];
  auto* task_head = head;
  auto* bra_head = head;
  const void *primitive_pairs=nullptr, *primitive_pair_offsets=nullptr,
      *ao_coefficients=nullptr, *atom_positions=nullptr, *schwarz_bounds=nullptr,
      *density=nullptr, *fock=nullptr;
""".replace("HIGH_CLASS", str(high_class))
        + queue
        + r"""
  if (threadIdx.x == 0) ++fallthrough;
}
int main() {
  const unsigned high_class = HIGH_CLASS, low_class = LOW_CLASS;
  const unsigned stride = topology.batch_size + 1;
  topology.pair_class_offsets.resize(10 * stride);
  for (unsigned code = 0; code < 10; ++code) {
    for (int system = 0; system < topology.batch_size; ++system) {
      topology.pair_class_offsets[code * stride + system] = topology.pair_order.size();
      if (code != high_class && code != low_class) continue;
      const unsigned count = system == 4 && code == low_class ? 35 : (system + code) % 4;
      for (unsigned local = 0; local < count; ++local) {
        topology.pair_order.push_back(topology.shell_pair_bounds.size());
        topology.shell_pair_bounds.push_back(1.0 / (local + 1));
        topology.shell_pair_systems.push_back(system);
      }
    }
    topology.pair_class_offsets[code * stride + topology.batch_size] = topology.pair_order.size();
  }
  for (double tolerance : {2.0, 0.0, 0.3, 0.0}) {
    std::vector<std::pair<unsigned, unsigned>> expected;
    for (int system = 0; system < topology.batch_size; ++system) {
      for (unsigned bra_index = topology.pair_class_offsets[high_class * stride + system];
           bra_index < topology.pair_class_offsets[high_class * stride + system + 1]; ++bra_index) {
        for (unsigned ket_index = topology.pair_class_offsets[low_class * stride + system];
             ket_index < topology.pair_class_offsets[low_class * stride + system + 1]; ++ket_index) {
          const auto bra = topology.pair_order[bra_index], ket = topology.pair_order[ket_index];
          if ((high_class != low_class || bra >= ket) &&
              topology.shell_pair_bounds[bra] * topology.shell_pair_bounds[ket] >= tolerance &&
              (3 * bra + ket) % 17 == 0) expected.emplace_back(bra, ket);
        }
      }
    }
    actual.clear(); admitted = 0; fallthrough = 0;
    std::uint32_t head = 0;
    std::vector<std::thread> threads;
    for (unsigned index = 0; index < 64; ++index)
      threads.emplace_back([&,index] { threadIdx.x = index; worker<false>(&head,tolerance); });
    for (auto& thread : threads) thread.join();
    std::sort(expected.begin(),expected.end()); std::sort(actual.begin(),actual.end());
    assert(actual == expected && admitted == expected.size() && fallthrough == 0);
  }
  for (unsigned mode = 0; mode < 4; ++mode) {
    topology.fock_consumer = mode == 1 ? Consumer::Coulomb : Consumer::Exchange;
    topology.shell_pair_density_bounds = mode == 2 ? nullptr : &topology;
    topology.system_pair_density_bounds = mode == 3 ? nullptr : &topology;
    actual.clear(); admitted = 0; fallthrough = 0;
    std::uint32_t head = 0;
    std::vector<std::thread> threads;
    for (unsigned index = 0; index < 64; ++index)
      threads.emplace_back([&,index] {
        threadIdx.x = index;
        if (mode == 0) worker<true>(&head,0.3); else worker<false>(&head,0.3);
      });
    for (auto& thread : threads) thread.join();
    assert(actual.empty() && admitted == 0 && fallthrough == 1 && head == 0);
  }
}
""".replace("HIGH_CLASS", str(high_class)).replace("LOW_CLASS", str(low_class))
    )
    executable = tmp_path / "admission"
    required_native_cxx.build_executable(
        [driver],
        executable,
        compile_args=("-std=c++20", "-O2", "-pthread"),
        link_args=("-pthread",),
    )
    subprocess.run([str(executable)], check=True, timeout=30)
