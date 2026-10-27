"""Execute the response planner at capacity cliffs and mandatory-storage limits."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx


def test_screening_headroom_preserves_bounded_response(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    source = tmp_path / "response_budget.cpp"
    source.write_text(r"""
#include "scf/df_response_tile_budget.hpp"
#include <algorithm>
#include <cassert>
#include <limits>

int main() {
  using generativeqc::scf::df_response_tile_with_headroom;
  // Independently enumerate valid panels for linear and occupied scratch.
  // Optional storage cannot displace the mandatory minimum, including at
  // exact byte boundaries and when all panel widths share fixed scratch.
  for (bool occupied : {false, true}) {
    auto bytes = [=](std::size_t q) { return 113 + std::max(occupied ? 97UL : 0UL, 17*q); };
    for (std::size_t cap = 1; cap <= 20; ++cap) {
      for (std::size_t available = bytes(1); available <= bytes(20)+41; ++available) {
        for (std::size_t extra : {0U, 1U, 16U, 17U, 41U, 512U}) {
          std::size_t expected = cap;
          if (extra && extra <= available && bytes(1) <= available-extra) {
            expected = 1;
            for (std::size_t q = 1; q <= cap; ++q)
              if (bytes(q) <= available-extra) expected = q;
          }
          const auto actual = df_response_tile_with_headroom(cap, available, extra, bytes);
          assert(actual == expected && actual <= cap && actual > 0);
        }
      }
    }
  }
  // Exercise the observed PBE0-96 scale without overflowing at size_t limits.
  auto workspace = [](std::size_t q) { return 4*3616UL*3616*8 + (3+2*q)*768UL*768*8 + 2*3616UL*8; };
  const auto room = 4325728307UL - 768UL*768*8;
  const auto q = df_response_tile_with_headroom(412, room, 1189120, workspace);
  assert(q == 411 && workspace(q)+1189120 <= room);
  auto identity = [](std::size_t q) { return q; };
  const auto maximum = std::numeric_limits<std::size_t>::max();
  assert(df_response_tile_with_headroom(maximum, maximum, 1, identity) == maximum-1);
  assert(df_response_tile_with_headroom(1, 0, maximum, identity) == 1);
  assert(df_response_tile_with_headroom(0, 0, 0, identity) == 0);
}
""")
    binary = tmp_path / "response_budget"
    root = Path(__file__).resolve().parents[2]
    native_cxx.build_executable(
        [source],
        binary,
        compile_args=(
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(root / "src"),
        ),
    )
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=30)
