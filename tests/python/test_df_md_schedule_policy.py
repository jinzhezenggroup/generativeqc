"""Executable work, device-domain and borrowed-capacity schedule contracts."""

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx


def test_charge_schedule_bounds(
    tmp_path: Path, required_native_cxx: "NativeCxx"
) -> None:
    """Automatic splitting is bounded; explicit rollback and small work persist."""
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "charge_schedule.cpp"
    source.write_text(r"""
#include "scf/df_md_schedule_policy.hpp"
#include <cassert>
#include <limits>

int main() {
  using generativeqc::scf::df_md_charge_splits;
  constexpr auto maximum = std::numeric_limits<std::size_t>::max();
  // The automatic domain is explicit; unqualified devices and short classes
  // cannot become partitioned merely because abundant scratch is available.
  for (unsigned architecture : {0U, 80U, 90U, 100U, 121U})
    assert(df_md_charge_splits(0, architecture, maximum, maximum) == 1);
  for (std::size_t sources : {0U, 1U, 8191U})
    assert(df_md_charge_splits(0, 120, sources, maximum) == 1);
  assert(df_md_charge_splits(0, 120, 8192, maximum) > 1);
  for (auto sources : {std::size_t{8192}, maximum}) {
    for (unsigned requested : {0U, 1U, 3U, 8U, 32U, 64U}) {
      for (std::size_t capacity = 0; capacity <= 65; ++capacity) {
        const auto splits = df_md_charge_splits(requested, 120, sources, capacity);
        assert(splits >= 1 && splits <= 64);
        if (splits > 1) assert(splits <= capacity);
        if (capacity <= 1 || requested == 1) assert(splits == 1);
        if (requested && capacity >= requested) assert(splits == requested);
      }
    }
  }
}
""")
    executable = required_native_cxx.build_executable(
        [source],
        tmp_path / "charge_schedule",
        compile_args=("-std=c++20", "-I" + str(root / "src")),
    )
    subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=30
    )
