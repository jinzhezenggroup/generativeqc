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
  using generativeqc::scf::df_recycled_projection_preferred;
  // Recycled storage cannot displace retained factors or authorize a scratch
  // interval too small for the algorithm. It is optional when owned work fits.
  assert(!df_recycled_projection_preferred(100, 101, 1, 20, 80));
  assert(!df_recycled_projection_preferred(100, 60, 50, 60, 49));
  assert(!df_recycled_projection_preferred(100, 60, 20, 40, 80));
  assert(df_recycled_projection_preferred(100, 60, 20, 41, 80));
  assert(df_recycled_projection_preferred(100, 60, 50, 60, 50));
  // Falling short of the preferred 512-column panel does not justify
  // replacing a viable 499-column owned panel with 25 recycled columns.
  assert(!df_recycled_projection_preferred(600, 101, 1, 512, 25));
  assert(!df_recycled_projection_preferred(600, 101, 1, 512, 498));
  assert(df_recycled_projection_preferred(600, 101, 1, 512, 499));
  assert(df_recycled_projection_preferred(600, 101, 1, 512, 512));
  assert(!df_recycled_projection_preferred(maximum, maximum-20, 20, 20, 80));
  assert(df_recycled_projection_preferred(maximum, maximum-20, 20, 21, 80));
  using generativeqc::scf::df_occupied_response_workspace;
  assert(!df_occupied_response_workspace(0,1,1).elements);
  assert(!df_occupied_response_workspace(1,maximum,1).elements);
  assert(!df_occupied_response_workspace(maximum,1,1).elements);
  assert(!df_occupied_response_workspace(1,1,maximum).elements);
  // Distinct live intervals must fit the declared fixed allocation. In the
  // observed nonlocal replay, omitting dead metric scratch must admit the
  // retained factors without changing their representation or dropping them.
  for (auto n : {1U, 7U, 768U}) for (auto a : {1U, 11U, 3616U})
    for (auto terms : {1U, 2U, 3U}) for (bool compact : {false, true})
      for (bool spectral : {false, true}) {
        const auto plan = df_occupied_response_workspace(n,a,terms,compact,spectral);
        assert(plan.metric_adjoint+a*a <= plan.ao_temporary);
        assert(plan.ao_temporary+3UL*n*n <= plan.charges);
        assert(plan.charges+terms*a <= plan.potentials);
        assert(plan.potentials+terms*a == plan.elements);
        if (compact && spectral) assert(a*a <= plan.metric_adjoint);
      }
  using generativeqc::scf::df_spectral_occupied_workspace;
  for (const auto bad : {std::size_t{0}, maximum}) {
    assert(!df_spectral_occupied_workspace(bad,11,2,3));
    assert(!df_spectral_occupied_workspace(4,bad,2,3));
    assert(!df_spectral_occupied_workspace(4,11,bad,3));
  }
  assert(!df_spectral_occupied_workspace(4,11,5,3));
  assert(!df_spectral_occupied_workspace(4,11,2,12));
  using generativeqc::scf::df_source_response_fixed_workspace;
  using generativeqc::scf::df_response_optional_storage_fits;
  assert(!df_source_response_fixed_workspace(0,384,1,true));
  assert(!df_source_response_fixed_workspace(96,0,1,true));
  assert(!df_source_response_fixed_workspace(96,384,0,true));
  assert(!df_source_response_fixed_workspace(maximum,384,1,true));
  assert(!df_source_response_fixed_workspace(96,maximum,1,true));
  assert(!df_source_response_fixed_workspace(96,384,maximum,true));
  assert(!df_source_response_fixed_workspace(96,384,1,false,20));
  assert(!df_source_response_fixed_workspace(96,384,2,true,20));
  assert(!df_source_response_fixed_workspace(96,384,1,true,97));
  assert(!df_source_response_fixed_workspace(1,1,maximum/3,false));
  const auto large_n = (std::size_t{1} << ((std::numeric_limits<std::size_t>::digits-2)/2))-1;
  assert(!df_source_response_fixed_workspace(large_n,1,1,true));
  // Optional shell metadata must coexist with the selected response layout,
  // including truncated occupied factors or the complete dense spectral root.
  // A cap that admits only generic response must retain that route. Test every
  // byte of the generic-only interval, then the exact shell-admission boundary.
  constexpr std::size_t n = 96, a = 384, rank = 20;
  constexpr std::size_t shell_bytes = (96+384)*12+16;
  constexpr std::size_t coordinates_output = 6*32*sizeof(double);
  for (bool truncated : {false,true}) for (bool occupied : {false,true}) {
    if (occupied && !truncated) continue;
    const auto fixed = df_source_response_fixed_workspace(n,a,1,truncated,occupied ? rank : 0);
    const auto expected = occupied ? 4*a*a+2*a*rank*rank+2*n*n+n*rank
                                  : 4*a*a+4*n*n+2*a+(truncated ? a*n*n : 0);
    assert(fixed == expected);
    const auto generic_bytes = coordinates_output+(fixed+2*n*n)*sizeof(double);
    const auto shell_minimum = generic_bytes+shell_bytes;
    for (auto cap = generic_bytes; cap < shell_minimum; ++cap) {
      assert(generic_bytes <= cap);
      assert(!df_response_optional_storage_fits(generic_bytes,shell_bytes,cap,maximum,maximum));
      assert(!df_response_optional_storage_fits(generic_bytes,shell_bytes,maximum,cap,maximum));
      assert(!df_response_optional_storage_fits(generic_bytes,shell_bytes,maximum,maximum,cap));
    }
    assert(df_response_optional_storage_fits(generic_bytes,shell_bytes,
                                            shell_minimum,shell_minimum,shell_minimum));
    if (occupied) {
      assert(generic_bytes == 7488000 && shell_minimum == 7493776);
      assert(df_spectral_occupied_workspace(n,a,rank,2)+n*n == fixed+2*n*n);
      // The old full-rank estimate incorrectly admitted shell metadata here.
      const auto old = df_source_response_fixed_workspace(n,a,1,false);
      assert(coordinates_output+shell_bytes+(old+2*n*n)*sizeof(double) < generic_bytes);
    }
  }
  assert(!df_response_optional_storage_fits(-1,1,maximum,maximum,maximum));
  assert(!df_response_optional_storage_fits(1,-1,maximum,maximum,maximum));
  assert(!df_response_optional_storage_fits(std::numeric_limits<long double>::infinity(),
                                           1,maximum,maximum,maximum));
  // The actual A100 LR response ceiling must cover the declared compact
  // tensors, density upload and a 64-Q AO output panel. Dense spectral Z
  // alone exceeds that ceiling, independent of how its panels are tiled.
  const auto projected = df_spectral_occupied_workspace(768,3616,160,64);
  assert(projected && 8*(projected+768UL*768) < 2549049300UL);
  assert(8*3616UL*768*768 > 2549049300UL);
  assert(df_spectral_occupied_workspace(768,3616,160,65)-projected == 768UL*768);
  const auto general = df_occupied_response_workspace(768,3616,1);
  const auto fitted = df_occupied_response_workspace(768,3616,1,true,false);
  const auto spectral = df_occupied_response_workspace(768,3616,1,true,true);
  const auto factors = 3616UL*160*160*8;
  const auto density = 768UL*768*8;
  assert(factors+general.elements*8+density > 954531840);
  assert(factors+fitted.elements*8+density < 954531840);
  assert(fitted.elements < spectral.elements && spectral.elements < general.elements);
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
