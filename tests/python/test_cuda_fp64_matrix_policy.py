"""Optional FP64 matrix schedules must preserve old and unknown GPU routes."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx


def test_fp64_matrix_capability_requires_architecture_and_throughput(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """Full scalar FP64 alone cannot qualify a matrix-accelerated schedule."""
    unit = tmp_path / "capability.cpp"
    unit.write_text(
        r"""
#include <iostream>
#include "runtime/cuda_fp64_matrix_policy.hpp"
int main() {
  int architecture, ratio;
  while (std::cin >> architecture >> ratio)
    std::cout << generativeqc::runtime::cuda_fp64_matrix_acceleration_supported(
                     architecture, ratio) << "\n";
}
"""
    )
    executable = required_native_cxx.build_executable(
        [unit],
        tmp_path / "capability",
        compile_args=(
            "-std=c++20",
            "-I" + str(Path(__file__).resolve().parents[2] / "src"),
        ),
    )
    # Older full-FP64 devices, unknown facts, and low-FP64 newer devices all
    # retain their original schedules. Neither product names nor environment
    # overrides may turn these hardware facts into a positive qualification.
    cases = {
        (0, 0): False,
        (0, 2): False,
        (-1, 2): False,
        (6, 2): False,
        (7, 2): False,
        (8, 0): False,
        (8, -1): False,
        (8, 1): True,
        (8, 2): True,
        (8, 3): False,
        (8, 64): False,
        (9, 2): True,
        (12, 64): False,
        (12, 0): False,
    }
    result = subprocess.run(
        [str(executable)],
        input="".join(f"{major} {ratio}\n" for major, ratio in cases),
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    assert [bool(int(value)) for value in result.stdout.split()] == list(cases.values())
