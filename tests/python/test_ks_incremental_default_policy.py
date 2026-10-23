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


def test_automatic_cold_promotion_does_not_override_explicit_controls() -> None:
    method = (ROOT / "src/methods/dft_method_preparation.cpp").read_text()
    cuda = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    assert 'std::getenv("GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK")' in method
    assert 'std::getenv("GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK")' in method
    assert "options.incremental_direct_jk_auto =" in method
    assert "incremental_policy_options.incremental_direct_jk |= auto_incremental;" in cuda
    assert "!runtime::active_device_resource_ledger" in cuda
    assert "(!options.incremental_direct_jk_auto || input == nullptr)" in cuda
    assert "!ordinary_incremental_warm" in cuda
