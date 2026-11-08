"""Execute the XC-only handoff's exact-source ownership without CUDA work."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from _cpp_source_support import cpp_function_definition

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_occupied_handoff_preserves_unchanged_exact_source(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """No new ERI allocation/production for an unchanged or absent FP64 lease.

    A true precision transition still requires a strict replacement. The native
    owner guard executes with fallible producer doubles, not a second policy.
    """
    owner = cpp_function_definition(
        (ROOT / "src/dft/cuda_ks.cpp").read_text(), "finish_legacy"
    )
    start = owner.index("// Mixed J needs a new strict source lease;")
    end = owner.index("} else if (pending_nonlocal_seed", start)
    driver = tmp_path / "handoff.cpp"
    driver.write_text(
        r"""
#include <cassert>
#include <memory>
#include <initializer_list>
struct Source {
  bool live, mixed;
  bool active() const { return live; }
  bool mixed_recurrence() const { return mixed; }
};
struct Owner {
  std::unique_ptr<Source> phase_source;
  bool phase_source_cold = true;
  void prepare_phase_source(bool) {
    auto replacement = std::make_unique<Source>(Source{true, false});
    phase_source = std::move(replacement);
  }
  void handoff(bool enter_occupied_audit) {
"""
        + owner[start:end]
        + r"""
  }
};
int main() {
  for (bool occupied : {false, true})
    for (bool present : {false, true})
      for (bool active : {false, true})
        for (bool mixed : {false, true}) {
          Owner owner;
          if (present) owner.phase_source = std::make_unique<Source>(Source{active, mixed});
          const auto* original = owner.phase_source.get();
          owner.handoff(occupied);
          if (occupied && !(present && active && mixed)) {
            assert(owner.phase_source.get() == original);
          } else {
            assert(owner.phase_source && owner.phase_source->active());
            assert(!owner.phase_source->mixed_recurrence());
            assert(owner.phase_source.get() != original);
          }
        }
}
"""
    )
    binary = native_cxx.build_executable(
        [driver], tmp_path / "handoff", compile_args=["-std=c++20"]
    )
    subprocess.run([binary], check=True, timeout=10)
