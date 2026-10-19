"""Host arithmetic checks for the real ordered adapter; not CUDA qualification.

Public launch admission is exercised by the required no-GPU native driver in
tests/native/test_ordered_gram_host_admission.cpp and its integer-oracle runner.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from tools.generate_scf_array_native import diis_cuda_header

if TYPE_CHECKING:
    from conftest import NativeCxx


ROOT = Path(__file__).resolve().parents[2]


def test_ordered_ring_adapter_lifetime_and_work(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    # Every lane writes disjoint cache elements. Sequential host invocation
    # validates that arithmetic/indexing contract, but cannot validate CUDA
    # execution, warp synchronization, launch routing or endpoint performance.
    (tmp_path / "cuda_runtime.h").write_text(
        "#pragma once\n#define __device__\n"
        "struct Lane { unsigned x; }; inline Lane threadIdx{};\n"
        "inline void __syncwarp() {}\n"
    )
    (tmp_path / "generated_scf_diis_cuda.cuh").write_text(diis_cuda_header())
    source = tmp_path / "ordered_ring_probe.cpp"
    source.write_text(
        r"""
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include "tensor/cuda_ring_gram.cuh"

bool same(double a, double b) {
  return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

void one_case(unsigned capacity, std::size_t elements, bool normalized) {
  using generativeqc::tensor::refresh_ring_gram;
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> history(capacity * elements, nan);
  // Include an inactive neighboring system which must remain untouched.
  std::vector<double> gram(std::size_t{2} * capacity * capacity, nan);
  generativeqc::tensor::RingGramWork work{};
  unsigned count = 0, head = 0;
  unsigned long long expected_dots = 0;
  for (unsigned step = 0; step < 3 * capacity + 19; ++step) {
    if (step == capacity + 7 || step == 2 * capacity + 11) {
      count = head = 0;  // Deliberately leave cache and history uncleared.
    }
    const auto before = gram;
    const auto previous_count = count;
    const auto inserted = head;
    for (std::size_t element = 0; element < elements; ++element)
      history[std::size_t{inserted} * elements + element] =
          std::sin(0.013 * (step + 1) * (element + 1)) +
          std::cos(0.021 * (step + 3) * (element + 1));
    count = std::min(count + 1, capacity);
    head = (head + 1) % capacity;
    const auto first = normalized ? (head + capacity - count) % capacity : 0;
    if (count >= 2) {
      for (unsigned lane = 0; lane < 32; ++lane) {
        threadIdx.x = lane;
        refresh_ring_gram(history.data(), elements, capacity, inserted, first, count,
                          previous_count == 1, gram.data(), &work);
      }
      expected_dots += count + (previous_count == 1);
    }
    const auto live = [&](unsigned slot) { return (slot + capacity - first) % capacity < count; };
    for (unsigned row = 0; row < capacity; ++row)
      for (unsigned column = 0; column < capacity; ++column) {
        const auto index = std::size_t{row} * capacity + column;
        const bool changed = count >= 2 &&
            ((row == inserted && live(column)) || (column == inserted && live(row)) ||
             (previous_count == 1 && row == first && column == first));
        if (!changed && !same(gram[index], before[index]))
          throw std::runtime_error("old-old or unused cache entry was modified");
        if (count < 2 || !live(row) || !live(column)) continue;
        long double oracle = 0;
        double ordered = 0;
        for (std::size_t element = 0; element < elements; ++element) {
          const auto a = history[std::size_t{row} * elements + element];
          const auto b = history[std::size_t{column} * elements + element];
          oracle += static_cast<long double>(a) * b;
          ordered += a * b;
        }
        if (!same(gram[index], ordered) || !std::isfinite(gram[index]) ||
            std::abs(gram[index] - oracle) > 3e-12L * (1 + std::abs(oracle)))
          throw std::runtime_error("ordered cache differs from independent oracle");
      }
    for (std::size_t i = std::size_t{capacity} * capacity; i < gram.size(); ++i)
      if (!std::isnan(gram[i])) throw std::runtime_error("inactive cache mutated");
    if (work.dots != expected_dots || work.vector_elements != expected_dots * elements)
      throw std::runtime_error("lazy work counters differ");
    // Simulate solver retirement of the oldest entries, retaining at least 2.
    if (normalized && count > 2 && step % 5 == 3) count = 2;
  }
}

int main() {
  unsigned cases = 0;
  for (unsigned capacity = 2; capacity <= 64; ++capacity)
    for (std::size_t elements : {1U, 9U, 37U})
      for (bool normalized : {false, true}) {
        one_case(capacity, elements, normalized);
        ++cases;
      }
  std::cout << cases << " ordered adapter cases passed\n";
}
"""
    )
    executable = tmp_path / "ordered_ring_probe"
    required_native_cxx.build_executable(
        [source],
        executable,
        compile_args=(
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            "-I" + str(tmp_path),
            "-I" + str(ROOT / "src"),
        ),
    )
    result = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=60
    )
    assert result.stdout == "378 ordered adapter cases passed\n"
