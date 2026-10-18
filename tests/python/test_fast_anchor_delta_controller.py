"""Execute the Fast anchor admission contract without requiring a CUDA device."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_fast_anchor_admission(required_native_cxx: NativeCxx, tmp_path: Path) -> None:
    """Separate measured default profitability from scientific opt-in eligibility."""
    source = tmp_path / "admission.cpp"
    source.write_text(r"""
#include <cassert>
#include "dft/application_precision.hpp"
int main() {
  using generativeqc::dft::admit_fast_anchor_delta;
  using generativeqc::dft::default_fast_anchor_delta;
  using generativeqc::dft::SemilocalFamily;
  using generativeqc::dft::semilocal_family_code;
  const auto wb = semilocal_family_code(SemilocalFamily::Wb97mv);
  for (const std::size_t aos : {0U, 24U, 96U, 191U, 769U, 1024U})
    assert(!default_fast_anchor_delta(aos));
  for (const std::size_t aos : {192U, 193U, 256U, 383U, 384U, 512U, 767U, 768U})
    assert(default_fast_anchor_delta(aos));
  assert(admit_fast_anchor_delta(true, true, wb, true, 1, true, false, 1));
  assert(!admit_fast_anchor_delta(false, true, wb, true, 1, true, false, 1));
  assert(!admit_fast_anchor_delta(true, false, wb, true, 1, true, false, 1));
  assert(!admit_fast_anchor_delta(true, true, wb, false, 1, true, false, 1));
  assert(!admit_fast_anchor_delta(true, true, wb, true, 2, true, false, 1));
  assert(!admit_fast_anchor_delta(true, true, wb, true, 1, false, false, 1));
  assert(!admit_fast_anchor_delta(true, true, wb, true, 1, true, true, 1));
  assert(!admit_fast_anchor_delta(true, true, wb, true, 1, true, false, 2));
  for (const auto family : {SemilocalFamily::Lda, SemilocalFamily::Pbe,
                            SemilocalFamily::R2scan, SemilocalFamily::B3lyp})
    assert(!admit_fast_anchor_delta(true, true, semilocal_family_code(family),
                                   true, 1, true, false, 1));
}
""")
    executable = tmp_path / "admission"
    required_native_cxx.build_executable(
        [source],
        executable,
        compile_args=("-std=c++20", f"-I{ROOT / 'src'}", f"-I{ROOT / 'include'}"),
    )
    subprocess.run([str(executable)], check=True, timeout=20)
