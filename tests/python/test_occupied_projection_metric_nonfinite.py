"""Execute actual native projection admission, including overflowing Gram inputs.

The one-column accepted projection has an independent analytic density. Compile
the real translation units rather than extracting a production function body or
substituting a test-only eigensolver. This is host validation, not a GPU test.
"""

import subprocess
import typing
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DRIVER = r"""
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include "integrals/s_integrals.hpp"
#include "scf/initial_guess/density.hpp"

int main(int argc, char** argv) {
  if (argc != 4) return 99;
  using generativeqc::scf::initial_guess::project_occupied_density;
  using Matrix = std::vector<double>;
  const double magnitude = std::strtod(argv[1], nullptr);
  const double sign = std::strtod(argv[2], nullptr);
  const bool invalid = std::atoi(argv[3]);
  generativeqc::integrals::IntegralData target;
  target.nbf = 1;
  target.overlap = {1};
  const Matrix orthogonalizer{1};
  const Matrix source_metric = invalid
      ? Matrix{1e308, -sign * 5e307, -sign * 5e307, 1e308}
      : Matrix{1, 0, 0, 1};
  const Matrix coefficients = invalid
      ? Matrix{magnitude, 0, sign * magnitude, 1}
      : Matrix{sign, 0, 0, 1};
  const Matrix cross = invalid
      ? Matrix{0.5 / magnitude, sign * 0.5 / magnitude}
      : Matrix{sign, 0};
  try {
    const auto result = project_occupied_density(target, orthogonalizer,
        source_metric, cross, 2, coefficients, 1, 2.0, 0.5);
    if (invalid) {
      std::cerr << "accepted nonfinite source Gram: "
                << result.source_metric_orthogonality_error;
      return 1;
    }
    if (std::abs(result.density[0] - 2) > 1e-12) return 2;
  } catch (const std::invalid_argument& error) {
    if (!invalid) { std::cerr << error.what(); return 3; }
    if (std::string(error.what()) != "source occupied metric contraction is non-finite") {
      std::cerr << error.what(); return 4;
    }
  }
}
"""


@pytest.fixture(scope="module")
def projection_probe(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: typing.Any
) -> Path:
    directory = tmp_path_factory.mktemp("projection-metric")
    source = directory / "probe.cpp"
    source.write_text(DRIVER)
    return native_cxx.build_executable(
        [
            source,
            ROOT / "src/scf/initial_guess/density.cpp",
            ROOT / "src/scf/reference/linalg.cpp",
        ],
        directory / "probe",
        compile_args=(
            "-std=c++20",
            "-O2",
            "-ffunction-sections",
            "-fdata-sections",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ),
        link_args=("-Wl,--gc-sections",),
    )


@pytest.mark.parametrize("sign", (-1, 1))
@pytest.mark.parametrize("magnitude,invalid", ((4, 1), (8, 1), (100, 1), (1, 0)))
def test_nonfinite_source_metric_cannot_pass_orthogonality(
    projection_probe: Path, magnitude: int, sign: int, invalid: int
) -> None:
    result = subprocess.run(
        [str(projection_probe), str(magnitude), str(sign), str(invalid)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
