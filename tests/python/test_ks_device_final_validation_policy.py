"""Execute the production host policy without another CUDA timing cohort."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_default_and_explicit_selectors_preserve_the_qualified_scope(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Cover size/spin/method/precision boundaries and strict opt-out parsing."""
    source = tmp_path / "policy.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include "dft/cuda_ks_final_validation_policy.hpp"

int main() {
  using namespace generativeqc::dft;
  const double exact_exchange_fraction = 0.25;
  const double restricted_coefficient = -exact_exchange_fraction / 2.0;
  assert(pbe0_rks_final_validation_composition(true, restricted_coefficient, 0.75, 1.0));
  assert(!pbe0_rks_final_validation_composition(true, exact_exchange_fraction, 0.75, 1.0));
  assert(!pbe0_rks_final_validation_composition(true, -exact_exchange_fraction, 0.75, 1.0));
  assert(!pbe0_rks_final_validation_composition(true, 0.0, 1.0, 1.0));
  assert(!pbe0_rks_final_validation_composition(false, restricted_coefficient, 0.75, 1.0));
  assert(!pbe0_rks_final_validation_composition(true, restricted_coefficient, 1.0, 1.0));
  assert(!pbe0_rks_final_validation_composition(true, restricted_coefficient, 0.75, 0.5));
  assert(!pbe0_rks_final_validation_composition(
      true, std::numeric_limits<double>::quiet_NaN(), 0.75, 1.0));
  for (std::size_t aos : {0U, 2U, 7U, 16U, 17U, 37U, 383U, 384U, 768U, 1024U}) {
    for (unsigned spins : {0U, 1U, 2U}) {
      for (bool direct_pbe0 : {false, true}) {
        for (bool full_precision : {false, true}) {
          const bool expected = aos >= 384 && spins == 1 && direct_pbe0 && full_precision;
          const bool eligible = device_final_validation_default_eligible(
              aos, spins, direct_pbe0, full_precision);
          assert(eligible == expected);
          assert(device_final_validation_requested(nullptr, eligible) == expected);
          assert(!device_final_validation_requested("0", eligible));
          assert(device_final_validation_requested("1", eligible));
          for (const char* invalid : {"", "auto", "2", "01", " 1", "true"}) {
            bool rejected = false;
            try { device_final_validation_requested(invalid, eligible); }
            catch (const std::invalid_argument&) { rejected = true; }
            assert(rejected);
          }
        }
      }
    }
  }
  assert(device_final_validation_default_eligible(
      std::numeric_limits<std::size_t>::max(), 1, true, true));
}
""",
        encoding="utf-8",
    )
    executable = native_cxx.build_executable(
        [source],
        tmp_path / "policy",
        compile_args=["-std=c++20", f"-I{ROOT / 'src'}"],
    )
    completed = subprocess.run(
        [executable], capture_output=True, text=True, check=False, timeout=10
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
