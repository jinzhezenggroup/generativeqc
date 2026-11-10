"""Host-only policy checks for the proposed GPU4PySCF-style RKS SCF default.

These tests do not qualify force accuracy or GPU runtime; the complete
original-grid/moved-force and README E+F campaigns remain separate gates.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx


def test_default_fullrange_rks_orbital_stopping_domain(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """Compile the actual C++ policy; no CUDA context or mock SCF solver."""
    source = tmp_path / "rks_orbital_policy.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <cmath>
#include <limits>
#include "dft/ks_stopping_policy.hpp"

int main() {
  using namespace generativeqc::dft;
  const auto pbe = semilocal_family_code(SemilocalFamily::Pbe);
  const auto b3lyp = semilocal_family_code(SemilocalFamily::B3lyp);
  const auto r2scan = semilocal_family_code(SemilocalFamily::R2scan);
  assert(early_final_probe_default_model(pbe, 1.0, 1.0, 0.0, true));
  assert(early_final_probe_default_model(pbe, 0.75, 1.0, -0.125, true));
  assert(early_final_probe_default_model(b3lyp, 1.0, 1.0, -0.1, true));
  assert(early_final_probe_default_model(r2scan, 1.0, 1.0, 0.0, true));
  assert(!early_final_probe_default_model(pbe, 1.0, 1.0, 0.0, false));

  // A standard, recognized full-range RKS owner uses the physical MO norm
  // independently of whether the public consumer requests E or E+F.
  assert(default_fullrange_ks_orbital_mode(true, true, false, false, false));
  for (unsigned mask = 1; mask < 32; ++mask) {
    const bool standard = (mask & 1) == 0;
    const bool model = (mask & 2) == 0;
    const bool nonlocal = (mask & 4) != 0;
    const bool incremental = (mask & 8) != 0;
    const bool reference = (mask & 16) != 0;
    assert(!default_fullrange_ks_orbital_mode(
        standard, model, nonlocal, incremental, reference));
  }

  // GPU4PySCF v1.9.0 tests |dE| < conv_tol and ||2 Cvir^T F Cocc||_F
  // < sqrt(conv_tol), not AO max or a separate density-change threshold.
  KsStoppingState state{9, false, true, true, 5e-10, 1e-4,
                        5e-6, 4e-6, 1e-9, 1e-5, 1e-9};
  assert(orbital_ks_converged(state, 3e-5));
  assert(!ordinary_ks_converged(state));
  assert(!orbital_ks_converged(state, 4e-5));
  assert(!orbital_ks_converged(state, std::sqrt(1e-9)));
  for (double bad : {-1.0, std::numeric_limits<double>::infinity(),
                     std::numeric_limits<double>::quiet_NaN()})
    assert(!orbital_ks_converged(state, bad));
  state.energy_change = 1e-9;
  assert(!orbital_ks_converged(state, 3e-5));
  state.energy_change = 5e-10;
  state.target_ready = false;
  assert(!orbital_ks_converged(state, 3e-5));
  state.target_ready = true;
  state.full_energy_history = false;
  assert(!orbital_ks_converged(state, 3e-5));
}
"""
    )
    root = Path(__file__).resolve().parents[2]
    binary = required_native_cxx.build_executable(
        [source],
        tmp_path / "rks-orbital-policy",
        compile_args=("-std=c++20", f"-I{root / 'src'}"),
    )
    subprocess.run([binary], check=True, capture_output=True, text=True, timeout=10)
