"""Opt-in mixed controls must preserve the original strict FP64 option domain."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from _cc_owner_test_support import compile_owner

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def option_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/cc/solver.cpp").read_text()
    start = source.index("void validate_options(const SolverOptions& o)")
    end = source.index("\nstd::size_t problem_host_bytes", start)
    folder = tmp_path_factory.mktemp("cc-option-validation")
    unit, binary = folder / "options.cpp", folder / "options"
    unit.write_text(
        '#include "cc/solver.hpp"\n'
        "#include <cmath>\n#include <limits>\n#include <stdexcept>\n"
        "#include <string>\nnamespace generativeqc::cc {\n"
        + source[start:end]
        + "\n}\n"
        + DRIVER
    )
    compile_owner(compiler, folder, [unit], binary)
    return binary


@pytest.mark.parametrize(
    "case,accepted",
    [
        ("fp64-default", True),
        ("fp64-upper-residual", True),
        ("fp64-unused-nan", True),
        ("fp64-unused-infinity", True),
        ("fp64-unused-zero", True),
        ("fp64-unused-cap-zero", True),
        ("fp64-residual-zero", False),
        ("fp64-residual-above-limit", False),
        ("mixed-default", True),
        ("mixed-upper-residual-decimal-switch", False),
        ("mixed-exact-lower-switch", True),
        ("mixed-below-lower-switch", False),
        ("mixed-above-lower-switch", True),
        ("mixed-exact-upper-switch", True),
        ("mixed-above-upper-switch", False),
        ("mixed-nan", False),
        ("mixed-infinity", False),
        ("mixed-zero", False),
        ("mixed-cap-zero", False),
        ("mixed-cap-one", True),
        ("mixed-residual-above-limit", False),
    ],
)
def test_original_fp64_and_strict_mixed_option_boundaries(
    option_probe: Path, case: str, accepted: bool
) -> None:
    result = subprocess.run(
        [str(option_probe), case],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == (0 if accepted else 1), result.stdout + result.stderr


DRIVER = r"""
int main(int argc,char** argv) {
  if(argc!=2) return 2;
  const std::string name=argv[1];
  generativeqc::cc::SolverOptions options;
  options.df_mixed_precision=name.starts_with("mixed-");
  if(name=="fp64-upper-residual" || name=="mixed-upper-residual-decimal-switch")
    options.residual_tolerance=1e-9;
  if(name=="mixed-upper-residual-decimal-switch")
    options.df_mixed_precision_switch_residual=1e-7;
  if(name.ends_with("-nan"))
    options.df_mixed_precision_switch_residual=std::numeric_limits<double>::quiet_NaN();
  if(name.ends_with("-infinity"))
    options.df_mixed_precision_switch_residual=std::numeric_limits<double>::infinity();
  if(name=="fp64-unused-zero" || name=="mixed-zero")
    options.df_mixed_precision_switch_residual=0;
  if(name.ends_with("-cap-zero")) options.df_mixed_precision_max_evaluations=0;
  if(name=="mixed-cap-one") options.df_mixed_precision_max_evaluations=1;
  if(name=="fp64-residual-zero") options.residual_tolerance=0;
  if(name.ends_with("-residual-above-limit"))
    options.residual_tolerance=std::nextafter(1e-9,1.0);
  if(name=="mixed-exact-lower-switch" || name=="mixed-below-lower-switch" ||
     name=="mixed-above-lower-switch") {
    options.residual_tolerance=1e-9;
    options.df_mixed_precision_switch_residual=100.0*options.residual_tolerance;
    if(name=="mixed-below-lower-switch")
      options.df_mixed_precision_switch_residual=
        std::nextafter(options.df_mixed_precision_switch_residual,0.0);
    if(name=="mixed-above-lower-switch")
      options.df_mixed_precision_switch_residual=
        std::nextafter(options.df_mixed_precision_switch_residual,1.0);
  }
  if(name=="mixed-exact-upper-switch" || name=="mixed-above-upper-switch") {
    options.df_mixed_precision_switch_residual=1e-2;
    if(name=="mixed-above-upper-switch")
      options.df_mixed_precision_switch_residual=std::nextafter(1e-2,1.0);
  }
  try { generativeqc::cc::validate_options(options); }
  catch(const std::invalid_argument&) { return 1; }
  return 0;
}
"""
