"""Check bounded class grouping preserves every admitted Direct force task."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx


def test_class_queue_is_a_bounded_permutation(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """Empty, partial and full packets retain each task exactly once."""
    source = tmp_path / "queue.cpp"
    source.write_text(
        r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <random>
#include "scf/cuda/direct_force_queue_order.hpp"
#include "scf/direct_task_layout.hpp"

using namespace generativeqc::scf::cuda_execution;
int main() {
  constexpr auto capacity = generativeqc::scf::detail::kBoundedDirectQueueCapacity;
  std::mt19937 generator(42);
  for (unsigned size = 0; size <= capacity; ++size) {
    for (unsigned trial = 0; trial < 20; ++trial) {
      std::uint32_t counts[kDirectSpdForceQueueClasses]{};
      std::array<unsigned, capacity> classes{}, output{};
      output.fill(capacity);
      for (unsigned task = 0; task < size; ++task) {
        classes[task] = trial == 0 ? 0 : trial == 1 ? 20 : generator() % 21;
        ++counts[classes[task]];
      }
      assert(direct_force_class_queue_offsets(counts) == size);
      for (unsigned task = 0; task < size; ++task) {
        const auto slot = counts[classes[task]]++;
        assert(slot < size && output[slot] == capacity);
        output[slot] = task;
      }
      for (unsigned slot = 1; slot < size; ++slot)
        assert(classes[output[slot - 1]] <= classes[output[slot]]);
      std::sort(output.begin(), output.begin() + size);
      for (unsigned slot = 0; slot < size; ++slot) assert(output[slot] == slot);
      for (unsigned slot = size; slot < output.size(); ++slot) assert(output[slot] == capacity);
    }
  }
}
"""
    )
    executable = required_native_cxx.build_executable(
        [source],
        tmp_path / "queue",
        compile_args=(
            "-std=c++20",
            "-I" + str(Path(__file__).resolve().parents[2] / "src"),
        ),
    )
    subprocess.run([str(executable)], check=True, capture_output=True, timeout=15)
