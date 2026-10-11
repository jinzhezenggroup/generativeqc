"""Execution-local physical build accounting fails closed across all attempts."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx


def test_target_fock_work_owns_every_started_attempt(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """Execute the real header, without extracting or fingerprinting its body."""
    source = tmp_path / "work.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include "scf/target_fock_work.hpp"
using generativeqc::scf::TargetFockWork;
int main() {
  TargetFockWork complete;
  assert(!complete.total());
  complete.begin_attempt();
  assert(!complete.total());
  complete.finish_attempt(0);
  assert(complete.total() == 0);
  complete.begin_attempt();
  assert(!complete.total());
  complete.finish_attempt(17);
  complete.begin_attempt();
  complete.finish_attempt(53);
  assert(complete.total() == 70);

  const auto limit = std::numeric_limits<std::uint64_t>::max();
  TargetFockWork boundary;
  boundary.begin_attempt();
  boundary.finish_attempt(limit - 53);
  boundary.begin_attempt();
  boundary.finish_attempt(53);
  assert(boundary.total() == limit);
  boundary.begin_attempt();
  boundary.finish_attempt(1);
  assert(!boundary.total());

  for (unsigned mode = 0; mode < 4; ++mode) {
    TargetFockWork invalid;
    if (mode != 2) invalid.begin_attempt();
    if (mode == 0) invalid.finish_attempt(std::nullopt);
    if (mode == 1) invalid.begin_attempt();
    if (mode == 2) invalid.finish_attempt(17);
    // Mode 3 models an exception with an uncompleted physical attempt.
    assert(!invalid.total());
    invalid.begin_attempt();
    invalid.finish_attempt(53);
    assert(!invalid.total());
  }
  TargetFockWork fresh;
  fresh.begin_attempt();
  fresh.finish_attempt(53);
  assert(fresh.total() == 53);
}
"""
    )
    root = Path(__file__).resolve().parents[2]
    binary = required_native_cxx.build_executable(
        [source],
        tmp_path / "work",
        compile_args=(
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I" + str(root / "src"),
        ),
    )
    subprocess.run([str(binary)], check=True, timeout=10)
