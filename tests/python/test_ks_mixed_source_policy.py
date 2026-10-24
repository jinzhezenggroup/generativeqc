"""Execute proposal-only source screening guards without a GPU."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_mixed_source_predicate(required_native_cxx: NativeCxx, tmp_path: Path) -> None:
    """Opt-out, unsupported owners and unscreened targets retain their policy."""
    source = tmp_path / "screening.cpp"
    source.write_text(r"""
#include <cassert>
#include <cmath>
#include <limits>
#include "dft/application_precision.hpp"
int main() {
  using generativeqc::dft::mixed_source_proposal_screening;
  using generativeqc::dft::default_mixed_source_proposal;
  for (const std::size_t aos : {0U, 24U, 63U, 769U, 1200U})
    assert(!default_mixed_source_proposal(aos, 12));
  for (const std::size_t aos : {64U, 96U, 384U, 768U}) {
    for (const std::size_t atoms : {2U, 12U, 48U, 96U})
      assert(default_mixed_source_proposal(aos, atoms));
    for (const std::size_t atoms : {0U, 1U, 97U})
      assert(!default_mixed_source_proposal(aos, atoms));
  }
  for (const double threshold : {0., 1e-14, 1e-12, 1e-10, 1e-8, 1e-7, 1e-6}) {
    assert(mixed_source_proposal_screening(false, true, threshold) == threshold);
    assert(mixed_source_proposal_screening(true, false, threshold) == threshold);
    assert(mixed_source_proposal_screening(false, false, threshold) == threshold);
    const auto expected = threshold > 0. && threshold < 1e-7 ? 1e-7 : threshold;
    assert(mixed_source_proposal_screening(true, true, threshold) == expected);
  }
  // Invalid input is not repaired by the proposal policy: its original
  // validation owner must see and reject it.
  assert(std::isnan(mixed_source_proposal_screening(
      true, true, std::numeric_limits<double>::quiet_NaN())));
  assert(mixed_source_proposal_screening(true, true, -1.) == -1.);
}
""")
    executable = required_native_cxx.build_executable(
        [source],
        tmp_path / "screening",
        compile_args=("-std=c++20", f"-I{ROOT / 'src'}", f"-I{ROOT / 'include'}"),
    )
    subprocess.run([str(executable)], check=True, timeout=20)
