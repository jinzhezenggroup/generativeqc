"""Independent feature and source-lifetime contracts for occupied XC."""

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from _cpp_source_support import cpp_function_definition, cpp_record_definition
from _cuda_rounding_support import DIRECTED_ROUNDING_HOST
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
        "#include <cstddef>\n#include <cstdint>\n#include <cassert>\n#include <initializer_list>\n"
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
  for (std::size_t active = 0; active < 130; ++active) {
    for (std::size_t rank = 0; rank <= 130; ++rank) {
      witness.occupations[0] = rank;
      for (std::size_t terms : {1U, 4U, 5U}) {
        const std::size_t jets = terms == 1 ? 1 : 4, work_jets = terms == 5 ? 4 : 1;
        witness.provisional = true;
        const bool beneficial = rank && active && rank <= (active - 1) / (terms == 4 ? 2 : 1);
        assert(witness.projection_fits(active, jets, work_jets) == beneficial);
        witness.provisional = false;
        assert(witness.projection_fits(active, jets, work_jets) ==
               bool(terms == 5 && rank && rank <= active / 2));
      }
    }
  }
  witness.provisional = true;
  witness.occupations[0] = std::size_t(-1) / 2;
  assert(witness.projection_fits(std::size_t(-1), 4, 1));
  ++witness.occupations[0];
  assert(!witness.projection_fits(std::size_t(-1), 4, 1));
  assert(!witness.projection_fits(std::size_t(-1), 10, 1));
  assert(!witness.projection_fits(std::size_t(-1), 4, 0));
}
"""
    )
    executable = native_cxx.build_executable(
        [source], tmp_path / "witness", compile_args=["-std=c++17", "-O2"]
    )
    subprocess.run([executable], check=True, timeout=10)


@pytest.mark.parametrize("terms", (1, 4, 5))
@pytest.mark.parametrize("provisional", (False, True))
def test_generated_occupied_features_match_independent_density_matrix(
    tmp_path: Path, native_cxx: "NativeCxx", terms: int, provisional: bool
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
        "bool __any_sync(unsigned, bool value) { return value; }\n"
        "using std::isfinite;\n"
        "double finite(double value, int* error, int code) {\n"
        "  if (!std::isfinite(value)) *error = code; return value;\n}\n"
        + "namespace generativeqc_grid_policy {\n"
        + emit_feature_policy(device=False)
        + "}\n"
        + DIRECTED_ROUNDING_HOST
        + cpp_record_definition(
            (ROOT / "src/dft/cuda_xc.hpp").read_text(), "CudaXcOccupiedDensity"
        )
        + ";\n"
        + cpp_function_definition(
            _emit_occupied_features(DEFAULT_XC_MATRIX_SCHEDULE),
            "occupied_feature_certificate",
        )
        + body
        + r"""
int main(int argc, char** argv) {
  const I rank = std::stoll(argv[1]), count = std::stoll(argv[2]);
  const I spins = std::stoll(argv[3]);
  const I active = std::stoll(argv[6]);
  const I terms = std::stoll(argv[7]), jets = terms == 1 ? 1 : 4;
  const I work_jets = terms == 5 ? 4 : 1;
  const bool provisional = std::stoi(argv[8]);
  std::vector<double> work(spins * work_jets * rank * count), features(spins * terms * count);
  std::vector<double> ao(jets * active * count), density(spins * active * active);
  std::vector<double> frame(spins * active * rank);
  std::size_t occupations[2]{};
  std::ifstream input(argv[4], std::ios::binary);
  input.read(reinterpret_cast<char*>(work.data()), work.size() * sizeof(double));
  input.read(reinterpret_cast<char*>(ao.data()), ao.size() * sizeof(double));
  input.read(reinterpret_cast<char*>(density.data()), density.size() * sizeof(double));
  input.read(reinterpret_cast<char*>(frame.data()), frame.size() * sizeof(double));
  input.read(reinterpret_cast<char*>(occupations), sizeof(occupations));
  // The frame ABI uses a full nao-by-nao stride between spins/columns.
  std::vector<double> coefficients(spins * active * active);
  for (I spin = 0; spin < spins; ++spin)
    for (I orbital = 0; orbital < rank; ++orbital)
      for (I row = 0; row < active; ++row)
        coefficients[spin*active*active+orbital*active+row] = frame[(spin*active+row)*rank+orbital];
  CudaXcOccupiedDensity witness{density.data(), coefficients.data(), std::size_t(active),
      std::size_t(spins), {occupations[0], occupations[1]}, 1, provisional};
  int error = 0;
  occupied_density_features_body<false>(work.data(), rank, count, spins, terms,
                                        features.data(), &error, density.data(), ao.data(),
                                        active, active, nullptr, nullptr, provisional,
                                        witness.coefficients, witness.occupations[0],
                                        witness.occupations[1]);
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
    cases = [
        (occupations, scale)
        for occupations in ((1,), (5,), (33,), (5, 3), (0, 7))
        for scale in (1.0, 1e-308, 1e90)
    ]
    for occupations, scale in cases:
        spins, rank, active, points = len(occupations), max(occupations), 41, 19
        jets, work_jets = (1 if terms == 1 else 4), (4 if terms == 5 else 1)
        ao = rng.normal(size=(jets, points, active)) * 0.1
        frames = rng.normal(size=(spins, active, rank)) * 0.1 * scale
        if scale == 1e-308:
            ao[1:] *= 1e307
        elif scale != 1.0:
            ao /= scale
        work = np.zeros((spins, work_jets, points, rank))
        matrices = np.zeros((spins, active, active))
        reference = np.zeros((spins, terms, points))
        for spin, occupied in enumerate(occupations):
            frame = frames[spin, :, :occupied]
            work[spin, :, :, :occupied] = ao[:work_jets] @ frame
            density = frame @ frame.T * (2.0 if spins == 1 else 1.0)
            matrices[spin] = density
            dense_work = ao @ density
            reference[spin, 0] = np.sum(ao[0] * dense_work[0], axis=-1)
            if terms >= 4:
                reference[spin, 1:4] = 2.0 * np.sum(ao[1:4] * dense_work[0], axis=-1)
            if terms == 5:
                reference[spin, 4] = 0.5 * np.sum(
                    ao[1:4] * dense_work[1:4], axis=(0, 2)
                )
        input_path, output_path = tmp_path / "work.bin", tmp_path / "features.bin"
        if scale != 1.0:
            work[:, 0] = np.nan
        else:
            work[:, 0, 0] = np.nan
        with input_path.open("wb") as stream:
            work.tofile(stream)
            ao.tofile(stream)
            matrices.tofile(stream)
            frames.tofile(stream)
            np.asarray((*occupations, 0)[:2], dtype=np.uintp).tofile(stream)
        subprocess.run(
            [
                executable,
                str(rank),
                str(points),
                str(spins),
                input_path,
                output_path,
                str(active),
                str(terms),
                str(int(provisional)),
            ],
            check=True,
            timeout=10,
        )
        actual = np.fromfile(output_path).reshape(reference.shape)
        np.testing.assert_allclose(actual, reference, rtol=5e-13, atol=5e-14)
