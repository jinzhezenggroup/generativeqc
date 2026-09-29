"""Execute the shared AO Schwarz gate against the historical rejection rule."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_ao_schwarz_preserves_ieee_rejection_semantics(tmp_path: Path) -> None:
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for AO Schwarz execution")

    # Compile the actual scalar helper and matrix index, without mocking the
    # scientific comparison. Other functions in this CUDA header need DeviceBatch.
    source = (ROOT / "src/scf/cuda/direct_screening.cuh").read_text(encoding="utf-8")
    signature = "__device__ __forceinline__ bool direct_ao_quartet_survives_schwarz("
    assert source.count(signature) == 1
    begin = source.index(signature)
    end = source.index("\n}", begin) + len("\n}")
    helper = source[begin:end]
    program = tmp_path / "schwarz.cpp"
    executable = tmp_path / "schwarz"
    program.write_text(
        "#include <array>\n#include <cmath>\n#include <iostream>\n"
        "#include <limits>\n#define __device__\n#define __forceinline__ inline\n"
        '#include "scf/cuda/matrix_index.cuh"\n'
        "namespace generativeqc::scf::cuda_execution {\n"
        + helper
        + "\n}\n"
        + r"""
int main() {
  using namespace generativeqc::scf::cuda_execution;
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double tiny = std::numeric_limits<double>::denorm_min();
  const double big = std::numeric_limits<double>::max();
  const std::array<double, 15> values = {
      -inf, -big, -1.0, -tiny, -0.0, 0.0, tiny, 0.5, 1.0,
      std::nextafter(1.0, 0.0), std::nextafter(1.0, inf), 2.0, big, inf, nan};
  const std::array<std::array<std::size_t, 4>, 2> quartets = {
      {{{0, 1, 2, 3}}, {{3, 0, 1, 2}}}};
  constexpr std::size_t n = 4;
  std::size_t checked = 0;
  for (std::size_t offset : {std::size_t{0}, std::size_t{7}}) {
    for (const auto& quartet : quartets) {
      for (double first : values) {
        for (double second : values) {
          for (double tolerance : values) {
            std::array<double, 32> bounds;
            bounds.fill(-777.0);
            bounds[offset + matrix_index(quartet[0], quartet[1], n)] = first;
            bounds[offset + matrix_index(quartet[2], quartet[3], n)] = second;
            // Independent oracle: exactly the pre-refactor rejection branch.
            bool expected = true;
            if (first * second < tolerance) expected = false;
            const bool actual = direct_ao_quartet_survives_schwarz(
                bounds.data(), offset, n, quartet[0], quartet[1], quartet[2],
                quartet[3], tolerance);
            if (actual != expected) {
              std::cerr << "mismatch: " << first << ' ' << second << ' '
                        << tolerance << '\n';
              return 1;
            }
            ++checked;
          }
        }
      }
    }
  }
  std::cout << checked << '\n';
}
""",
        encoding="utf-8",
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-fno-fast-math",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT / "src"),
            str(program),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = subprocess.run(
        [str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.stdout.strip() == "13500"
