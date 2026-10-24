"""Independent feature and source-lifetime contracts for occupied XC."""

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from _cpp_source_support import cpp_function_definition, cpp_record_definition
from generativeqc_compiler.dft.feature_policy import emit_feature_policy
from generativeqc_compiler.dft.xc_contraction_cuda import (
    DEFAULT_XC_MATRIX_SCHEDULE,
    _emit_occupied_features,
)

ROOT = Path(__file__).resolve().parents[2]

if TYPE_CHECKING:
    from conftest import NativeCxx


def test_occupied_density_witness_rejects_stale_and_incompatible_sources(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    header = (ROOT / "src/dft/cuda_xc.hpp").read_text()
    declaration = cpp_record_definition(header, "CudaXcOccupiedDensity")
    source = tmp_path / "witness.cpp"
    source.write_text(
        "#include <cstddef>\n#include <cstdint>\n#include <cassert>\n"
        + declaration
        + r""";
int main() {
  double density[100]{}, frame[100]{}, foreign[100]{};
  CudaXcOccupiedDensity witness{density, frame, 10, 1, {4, 0}, 7};
  assert(witness.matches(density, 10, 1, 7));
  assert(!witness.matches(foreign, 10, 1, 7));
  assert(!witness.matches(density, 9, 1, 7));
  assert(!witness.matches(density, 10, 2, 7));
  assert(!witness.matches(density, 10, 1, 8));
  witness.occupations[1] = 3;
  assert(!witness.matches(density, 10, 1, 7));
  witness.spins = 2;
  assert(witness.matches(density, 10, 2, 7));
  witness.occupations[0] = 0;
  assert(witness.matches(density, 10, 2, 7));
  witness.occupations[1] = 11;
  assert(!witness.matches(density, 10, 2, 7));
  witness.occupations[1] = 0;
  assert(!witness.matches(density, 10, 2, 7));
  witness.occupations[0] = 4;
  witness.coefficients = nullptr;
  assert(!witness.matches(density, 10, 2, 7));
  witness.coefficients = frame;
  witness.generation = 0;
  assert(!witness.matches(density, 10, 2, 0));
}
"""
    )
    executable = native_cxx.build_executable(
        [source], tmp_path / "witness", compile_args=["-std=c++17", "-O2"]
    )
    subprocess.run([executable], check=True, timeout=10)


def test_generated_occupied_features_match_independent_density_matrix(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    source = tmp_path / "features.cpp"
    body = cpp_function_definition(
        _emit_occupied_features(DEFAULT_XC_MATRIX_SCHEDULE),
        "occupied_density_features_body",
        include_template=True,
    )
    source.write_text(
        "#include <cmath>\n#include <cstddef>\n#include <cstdint>\n"
        "#include <fstream>\n#include <vector>\n#define __device__\n"
        "using I = std::int64_t;\n"
        "struct Dim { I x; };\n"
        "Dim threadIdx{0}, blockIdx{0}, blockDim{1}, gridDim{1};\n"
        "double __shfl_down_sync(unsigned, double value, int) { return value; }\n"
        "double finite(double value, int* error, int code) {\n"
        "  if (!std::isfinite(value)) *error = code; return value;\n}\n"
        + "namespace generativeqc_grid_policy {\n"
        + emit_feature_policy(device=False)
        + "}\n"
        + body
        + r"""
int main(int argc, char** argv) {
  const I rank = std::stoll(argv[1]), count = std::stoll(argv[2]);
  const I spins = std::stoll(argv[3]);
  std::vector<double> work(spins * 4 * rank * count), features(spins * 5 * count);
  std::ifstream(argv[4], std::ios::binary).read(
      reinterpret_cast<char*>(work.data()), work.size() * sizeof(double));
  int error = 0;
  occupied_density_features_body<false>(work.data(), rank, count, spins, 5,
                                        features.data(), &error);
  std::ofstream(argv[5], std::ios::binary).write(
      reinterpret_cast<const char*>(features.data()), features.size() * sizeof(double));
  return error;
}
"""
    )
    executable = native_cxx.build_executable(
        [source],
        tmp_path / "features",
        compile_args=["-std=c++17", "-O2", "-ffp-contract=off"],
    )
    rng = np.random.default_rng(20261017)
    for occupations in ((1,), (5,), (33,), (5, 3), (0, 7)):
        spins, rank, active, points = len(occupations), max(occupations), 41, 19
        ao = rng.normal(size=(4, points, active)) * 0.1
        frames = rng.normal(size=(spins, active, rank)) * 0.1
        work = np.zeros((spins, 4, points, rank))
        reference = np.zeros((spins, 5, points))
        for spin, occupied in enumerate(occupations):
            frame = frames[spin, :, :occupied]
            work[spin, :, :, :occupied] = ao @ frame
            density = frame @ frame.T * (2.0 if spins == 1 else 1.0)
            dense_work = ao @ density
            reference[spin, 0] = np.sum(ao[0] * dense_work[0], axis=-1)
            reference[spin, 1:4] = 2.0 * np.sum(ao[1:4] * dense_work[0], axis=-1)
            reference[spin, 4] = 0.5 * np.sum(ao[1:4] * dense_work[1:4], axis=(0, 2))
        input_path, output_path = tmp_path / "work.bin", tmp_path / "features.bin"
        work.tofile(input_path)
        subprocess.run(
            [executable, str(rank), str(points), str(spins), input_path, output_path],
            check=True,
            timeout=10,
        )
        actual = np.fromfile(output_path).reshape(reference.shape)
        np.testing.assert_allclose(actual, reference, rtol=5e-13, atol=5e-14)
