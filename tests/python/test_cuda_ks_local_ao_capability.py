"""Host regression for component-wise CUDA KS local-AO admission."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _guard() -> str:
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    start = source.index("    const char* ao_selection = std::getenv(")
    end = source.index("    constexpr std::size_t ao_map_host_budget", start)
    return source[start:end]


def test_local_ao_guard_depends_on_layout_capability_not_method_or_schedule(
    tmp_path: Path,
) -> None:
    guard = _guard()
    assert "cuda_xc_execution_capabilities(xc_layout).local_ao_selection" in guard
    assert "precision_schedule" not in guard
    assert "SemilocalFamily" not in guard

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    unit = tmp_path / "guard.cpp"
    executable = tmp_path / "guard"
    unit.write_text(
        r"""
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

struct Layout { bool selectable{}; };
struct Capabilities { bool local_ao_selection{}, mixed_density_contraction{}; };
Capabilities cuda_xc_execution_capabilities(const Layout& layout) {
  return {layout.selectable, false};
}

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  if (std::string(argv[1]) == "unset")
    unsetenv("GENERATIVEQC_CUDA_KS_ACTIVE_AO");
  else
    setenv("GENERATIVEQC_CUDA_KS_ACTIVE_AO", argv[1], 1);
  Layout xc_layout{std::string(argv[2]) == "capable"};
  const bool host_unfused = std::string(argv[3]) == "host";
  try {
"""
        + guard
        + r"""
    std::cout << "accepted";
  } catch (const std::invalid_argument&) {
    std::cout << "rejected";
  }
}
"""
    )
    built = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(unit),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )
    assert built.returncode == 0, built.stderr

    cases = {
        ("unset", "blocked", "host"): "accepted",
        ("0", "blocked", "host"): "accepted",
        ("1", "capable", "device"): "accepted",
        ("1", "blocked", "device"): "rejected",
        ("1", "capable", "host"): "rejected",
        ("yes", "capable", "device"): "rejected",
    }
    for args, expected in cases.items():
        result = subprocess.run(
            [str(executable), *args],
            capture_output=True,
            text=True,
            check=True,
            env=os.environ.copy(),
        )
        assert result.stdout == expected


def test_iteration_path_intersects_schedule_with_xc_density_capability() -> None:
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    start = source.index(
        "      const auto iteration_precision = resolve_cuda_ks_iteration_precision("
    )
    end = source.index("      // Provider selection stays inside", start)
    block = source[start:end]
    assert (
        "cuda_xc_execution_capabilities(xc_layout).mixed_density_contraction" in block
    )
    assert (
        "iteration_precision.uses_lower_precision(cuda_ks_precision_region::kCoulombJ)"
        in block
    )
    assert (
        "iteration_precision.uses_lower_precision(cuda_ks_precision_region::kDensityContraction)"
        in block
    )
