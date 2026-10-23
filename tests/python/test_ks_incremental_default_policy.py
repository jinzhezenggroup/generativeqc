"""Executable automatic cold Direct J/K admission contract, without a GPU."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_auto_cold_guard_truth_table(tmp_path: Path, native_cxx: NativeCxx) -> None:
    """Compile the production constexpr policy; every exclusion must refuse."""
    source = tmp_path / "incremental_auto.cpp"
    source.write_text(
        """
#include <array>
#include <cassert>
#include <cstddef>
#include "dft/ks_incremental_policy.hpp"
using generativeqc::dft::KsIncrementalColdAutoDomain;
using generativeqc::dft::admit_ks_incremental_cold_auto;
using generativeqc::dft::ks_incremental_solve_active;
int main() {
  constexpr KsIncrementalColdAutoDomain good{true, 768, 96, true, true, true, true, true};
  static_assert(admit_ks_incremental_cold_auto(good));
  for (std::size_t n : std::array<std::size_t, 5>{0, 1, 63, 769, 1200}) {
    auto probe = good; probe.public_aos = n;
    assert(!admit_ks_incremental_cold_auto(probe));
  }
  for (std::size_t n : std::array<std::size_t, 4>{64, 128, 384, 768}) {
    auto probe = good; probe.public_aos = n;
    assert(admit_ks_incremental_cold_auto(probe));
  }
  for (std::size_t atoms : std::array<std::size_t, 3>{0, 1, 97}) {
    auto probe = good; probe.atoms = atoms;
    assert(!admit_ks_incremental_cold_auto(probe));
  }
  for (int guard = 0; guard < 6; ++guard) {
    auto probe = good;
    switch (guard) {
      case 0: probe.requested = false; break;
      case 1: probe.exact_fp64_direct_hybrid = false; break;
      case 2: probe.standard_default_grid = false; break;
      case 3: probe.through_d_shells = false; break;
      case 4: probe.all_electron = false; break;
      case 5: probe.no_resource_ledger = false; break;
    }
    assert(!admit_ks_incremental_cold_auto(probe));
  }
  // Seed provenance is independent of storage representation. In particular,
  // automatic Huckel/MINAO cold guesses arrive as non-null density buffers.
  for (bool prepared : {false, true})
    for (bool automatic : {false, true})
      for (bool warm : {false, true}) {
        assert(ks_incremental_solve_active(prepared, automatic, warm, false, false)
               == (prepared && !warm));
        assert(ks_incremental_solve_active(prepared, automatic, warm, true, true)
               == (prepared && !warm));
        assert(ks_incremental_solve_active(prepared, automatic, warm, true, false)
               == (prepared && !warm && !automatic));
      }
}
""",
        encoding="utf-8",
    )
    executable = native_cxx.build_executable(
        [source],
        tmp_path / "incremental_auto",
        compile_args=("-std=c++20", f"-I{ROOT / 'src'}"),
    )
    result = subprocess.run(
        [executable], capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
