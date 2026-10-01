"""Bitwise host probe of the actual emitted CUDA population-update helper."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("compiler_name", ["g++", "clang++"])
def test_emitted_population_preserves_fma_and_signed_zero(
    tmp_path: Path, compiler_name: str
) -> None:
    compiler = shutil.which(compiler_name)
    if compiler is None:
        pytest.skip(f"{compiler_name} is unavailable")
    header = tmp_path / "electronic.cuh"
    subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "tools/generate_gfn2_electronic_cuda.py"),
            "--output",
            str(header),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    emitted = header.read_text()
    begin = emitted.index("__device__ inline bool gfn2_population_update_cuda_tensor(")
    end = emitted.index("\n}", begin) + 2
    helper = emitted[begin:end].replace("__device__ ", "", 1)
    probe = tmp_path / "population.cpp"
    probe.write_text(
        "#include <cmath>\n#include <cstdint>\n#include <cstring>\n"
        "#include <limits>\n#include <random>\n#include <iostream>\n"
        + helper
        + r"""
std::uint64_t bits(double value) {
  std::uint64_t result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}
int check(double density, double integral, double accumulator) {
  const double expected = std::fma(-density, integral, accumulator);
  double actual = 123.25;
  const bool success = gfn2_population_update_cuda_tensor(density, integral, accumulator, actual);
  const bool admitted = std::isfinite(density) && std::isfinite(integral) &&
                        std::isfinite(accumulator) && std::isfinite(expected);
  if (success != admitted) return 1;
  if (success && bits(actual) != bits(expected)) return 2;
  if (!success && actual != 123.25) return 3;
  return 0;
}
int main() {
  const double values[]{0.0, -0.0, 1.0, -1.0, 0x1p-1022, -0x1p-1022,
    std::numeric_limits<double>::denorm_min(), -std::numeric_limits<double>::denorm_min(),
    0x1.fffffffffffffp+1023, -0x1.fffffffffffffp+1023,
    std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::quiet_NaN()};
  for (double density : values) for (double integral : values) for (double accumulator : values)
    if (int status = check(density, integral, accumulator)) {
      std::cerr << density << ' ' << integral << ' ' << accumulator << '\n';
      return status;
    }
  std::mt19937_64 random(505);
  std::uniform_real_distribution<double> value(-100.0, 100.0);
  for (unsigned i = 0; i < 4096; ++i) {
    const double density = value(random), integral = value(random);
    // Exercise both general addition and cancellation around the unfused product.
    if (int status = check(density, integral, value(random))) return status;
    if (int status = check(density, integral, density * integral)) return status;
  }
}
"""
    )
    executable = tmp_path / "population"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(probe),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=10
    )
