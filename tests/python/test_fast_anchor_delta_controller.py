"""Execute shared FP64-anchor admission without requiring a CUDA device."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_fixed_anchor_admission(required_native_cxx: NativeCxx, tmp_path: Path) -> None:
    """Share scientific guards while keeping strict reuse explicitly selected."""
    source = tmp_path / "admission.cpp"
    source.write_text(r"""
#include <cassert>
#include "dft/application_precision.hpp"
int main() {
  using generativeqc::dft::admit_fixed_anchor_delta;
  using generativeqc::dft::admit_application_fock;
  using generativeqc::dft::default_fixed_anchor_delta;
  using generativeqc::dft::SemilocalFamily;
  using generativeqc::dft::semilocal_family_code;
  const auto wb = semilocal_family_code(SemilocalFamily::Wb97mv);
  for (const std::size_t aos : {0U, 24U, 96U, 191U})
    assert(!default_fixed_anchor_delta(true, true, aos));
  for (const std::size_t aos : {192U, 193U, 256U, 383U, 384U, 512U, 767U, 768U,
                               769U, 1024U, 1536U, 2048U, 4096U, 8192U})
    assert(default_fixed_anchor_delta(true, true, aos));
  for (const std::size_t aos : {769U, 1024U, 1536U, 2048U, 4096U, 8192U}) {
    const auto budget = admit_application_fock({1e-5, 1e-4, 3e-5}, aos, 192, 2,
                                                true, true, true);
    assert(budget && default_fixed_anchor_delta(budget, true, aos));
    assert(admit_fixed_anchor_delta(false, budget, true, wb, true, 1, true, false, 1));
  }
  for (const bool strict_fp64 : {false, true}) {
    for (const bool budget_fock : {false, true}) {
      for (const bool budget_stationarity : {false, true}) {
        const bool eligible = strict_fp64 || (budget_fock && budget_stationarity);
        assert(admit_fixed_anchor_delta(strict_fp64, budget_fock, budget_stationarity,
                                         wb, true, 1, true, false, 1) == eligible);
        assert(default_fixed_anchor_delta(budget_fock, budget_stationarity, 512) ==
               (budget_fock && budget_stationarity));
      }
    }
    for (unsigned guard = 0; guard < 5; ++guard)
      assert(!admit_fixed_anchor_delta(strict_fp64, true, true, wb, guard != 0,
                                       guard == 1 ? 2 : 1, guard != 2, guard == 3,
                                       guard == 4 ? 2 : 1));
    for (const auto family : {SemilocalFamily::Lda, SemilocalFamily::Pbe,
                              SemilocalFamily::R2scan, SemilocalFamily::B3lyp})
      assert(!admit_fixed_anchor_delta(strict_fp64, true, true, semilocal_family_code(family),
                                       true, 1, true, false, 1));
  }
}
""")
    executable = tmp_path / "admission"
    required_native_cxx.build_executable(
        [source],
        executable,
        compile_args=("-std=c++20", f"-I{ROOT / 'src'}", f"-I{ROOT / 'include'}"),
    )
    subprocess.run([str(executable)], check=True, timeout=20)
