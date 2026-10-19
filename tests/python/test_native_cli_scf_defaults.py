"""Compiled, GPU-free regression for the shared native CLI SCF defaults."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_cli_convergence_defaults_and_explicit_overrides(
    native_cxx: object, tmp_path: Path
) -> None:
    source = tmp_path / "cli_scf_defaults.cpp"
    source.write_text(
        r"""
#include "cli/options.hpp"

int main() {
  using generativeqc::cli::RunOptions;
  using generativeqc::cli::resolve_scf_convergence;
  RunOptions options;
  const auto standard = resolve_scf_convergence(options, false);
  if (standard.energy_tolerance != 1e-10 || standard.density_tolerance != 1e-8)
    return 1;

  const auto correlated = resolve_scf_convergence(options, true);
  if (correlated.energy_tolerance != 1e-12 || correlated.density_tolerance != 1e-10)
    return 2;

  // Explicit user tolerances override either profile without clamping.
  options.energy_tolerance = 3e-11;
  options.density_tolerance = 7e-9;
  const auto custom_mean_field = resolve_scf_convergence(options, false);
  const auto custom_correlated = resolve_scf_convergence(options, true);
  if (custom_mean_field.energy_tolerance != 3e-11 ||
      custom_mean_field.density_tolerance != 7e-9 ||
      custom_correlated.energy_tolerance != 3e-11 ||
      custom_correlated.density_tolerance != 7e-9)
    return 3;

  // An explicit tighter component does not change the other default.
  options.density_tolerance.reset();
  const auto mixed = resolve_scf_convergence(options, false);
  if (mixed.energy_tolerance != 3e-11 || mixed.density_tolerance != 1e-8)
    return 4;
  return 0;
}
""",
        encoding="utf-8",
    )
    binary = tmp_path / "cli_scf_defaults"
    native_cxx.build_executable(
        [source],
        binary,
        compile_args=(
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-I",
            str(ROOT / "src"),
            "-I",
            str(ROOT / "include"),
        ),
        compile_timeout=90,
    )
    subprocess.run([str(binary)], check=True, timeout=30)


@pytest.mark.parametrize(
    "filename",
    [
        "src/cli/main.cpp",
        "src/cli/geometry_command.cpp",
        "src/cli/resources.cpp",
    ],
)
def test_cli_run_optimize_and_resources_share_convergence_policy(filename: str) -> None:
    # These public paths must not quietly diverge in later refactors.
    source = (ROOT / filename).read_text(encoding="utf-8")
    assert "resolve_scf_convergence(" in source
