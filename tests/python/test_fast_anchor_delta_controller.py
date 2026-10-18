"""Execute the Fast anchor admission contract without requiring a CUDA device."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_fast_anchor_admission(required_native_cxx: NativeCxx, tmp_path: Path) -> None:
    """Exclude strict precision, unsupported models, and bounded resource owners."""
    source = tmp_path / "admission.cpp"
    source.write_text(r"""
#include <cassert>
#include "dft/application_precision.hpp"
int main() {
  using generativeqc::dft::admit_fast_anchor_delta;
  using generativeqc::dft::SemilocalFamily;
  using generativeqc::dft::semilocal_family_code;
  const auto wb = semilocal_family_code(SemilocalFamily::Wb97mv);
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
