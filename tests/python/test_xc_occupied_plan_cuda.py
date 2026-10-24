"""Native XC admission, compact maps and bounded borrowed-frame lifetimes."""

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TEST_OCCUPIED_XC") != "1",
    reason="requires explicit finite Slurm GPU allocation",
)
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_native_occupied_source_fallback_and_compact_domains(
    tmp_path: Path, native_cxx: "NativeCxx", spins: int, mapped: bool, compact: bool
) -> None:
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")
    library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
    cuda = Path(os.environ.get("CUDA_PATH", "/group/software/cuda-12.9.1"))
    source = tmp_path / "native_occupied.cpp"
    source.write_text(DRIVER)
    executable = native_cxx.build_executable(
        [source],
        tmp_path / "native_occupied",
        compile_args=[
            "-std=c++20",
            "-O2",
            "-DGENERATIVEQC_HAS_CUDA=1",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
            f"-I{cuda / 'include'}",
        ],
        link_args=[
            str(library),
            f"-L{cuda / 'lib64'}",
            "-lcudart",
            f"-Wl,-rpath,{library.parent}",
            f"-Wl,-rpath,{cuda / 'lib64'}",
        ],
    )
    subprocess.run(
        [executable, str(spins), str(int(mapped)), str(int(compact))],
        check=True,
        timeout=90,
    )


DRIVER = r"""
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "dft/cuda_xc.hpp"
#include "molecule/basis.hpp"

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool valid, const char* message) {
  if (!valid) throw std::runtime_error(message);
}
struct Stream {
  cudaStream_t value{};
  Stream() { check(cudaStreamCreate(&value)); }
  ~Stream() { if (value) cudaStreamDestroy(value); }
};
template<class Scalar> struct Buffer {
  Scalar* value{};
  explicit Buffer(size_t count) {
    check(cudaMalloc(reinterpret_cast<void**>(&value), count * sizeof(Scalar)));
  }
  ~Buffer() { if (value) cudaFree(value); }
};
int main(int argc, char** argv) {
  try {
    using namespace generativeqc;
    const size_t spins = std::stoul(argv[1]);
    const bool mapped = std::stoi(argv[2]), compact = std::stoi(argv[3]);
    core::System system;
    system.atoms = {{1, {0, 0, 0}}};
    system.multiplicity = 2;
    for (unsigned shell = 0; shell < 65; ++shell)
      system.shells.push_back({0, 0, {{0.2 + 0.1 * shell, 1.0}}});
    std::string detail;
    require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
             "invalid independent test basis");
    const dft::AoBasis basis(system);
    const dft::GridSpec spec{1, 4, 4, 4, 3, 1e-12};
    const dft::MolecularGrid grid(system, spec);
    auto layout = dft::cuda_xc_layout(basis, grid, 2, spins == 2, 32);
    dft::CudaXcAoTiles maps;
    if (mapped) {
      maps.derivative_order = 1;
      maps.offsets = {0, basis.nao, basis.nao + 32};
      for (size_t orbital = 0; orbital < basis.nao; ++orbital) maps.indices.push_back(orbital);
      for (size_t orbital = 0; orbital < 32; ++orbital) maps.indices.push_back(orbital);
      layout = dft::cuda_xc_local_ao_layout(layout, maps);
    }
    Stream stream;
    Buffer<unsigned char> arena(layout.device_bytes);
    const size_t matrix = basis.nao * basis.nao, elements = spins * matrix;
    Buffer<double> density(elements), coefficients(elements);
    std::vector<double> frames(elements), matrices(elements);
    for (size_t spin = 0; spin < spins; ++spin) {
      const size_t occupied = spin ? 17 : 33;
      for (size_t orbital = 0; orbital < occupied; ++orbital)
        for (size_t row = 0; row < basis.nao; ++row)
          frames[spin * matrix + orbital * basis.nao + row] =
              0.002 * (1 + (row + 3 * orbital + spin) % 7);
      for (size_t row = 0; row < basis.nao; ++row)
        for (size_t column = 0; column < basis.nao; ++column)
          for (size_t orbital = 0; orbital < occupied; ++orbital)
            matrices[spin * matrix + row + column * basis.nao] += (spins == 1 ? 2.0 : 1.0) *
                frames[spin * matrix + orbital * basis.nao + row] *
                frames[spin * matrix + orbital * basis.nao + column];
    }
    check(cudaMemcpyAsync(density.value, matrices.data(), elements * sizeof(double),
                           cudaMemcpyHostToDevice, stream.value));
    check(cudaMemcpyAsync(coefficients.value, frames.data(), elements * sizeof(double),
                           cudaMemcpyHostToDevice, stream.value));
    dft::CudaXcPlan owner(layout, basis.packed, grid.points(), grid.weights(), arena.value,
                         layout.device_bytes, stream.value, {}, mapped ? &maps : nullptr);
    if (compact) {
      owner.prepare_point_batches(3, 8 << 20, true, true);
      require(owner.point_batch_plan().compact_groups != 0, "compact route was not exercised");
    }
    using runtime::PrecisionPhase;
    owner.enqueue(density.value, elements, 1);
    const auto reference = owner.read_scalars(1);
    const auto potential = owner.download_potential(1);
    require(reference.error == 0, "reference matrix source failed");
    dft::CudaXcOccupiedDensity witness{density.value, coefficients.value, basis.nao, spins,
                                      {33, spins == 2 ? size_t(17) : size_t(0)}, 2};
    for (std::uint64_t generation = 2; generation < 9; ++generation) {
      auto view = witness;
      view.generation = generation;
      if (generation == 3) view.generation = generation - 1;
      if (generation == 4) view.density = coefficients.value;
      if (generation == 5) ++view.nao;
      if (generation == 6) view.occupations[0] = basis.nao + 1;
      if (generation == 7) view.spins = 3;
      if (generation == 8) view.coefficients = nullptr;
      owner.enqueue(density.value, elements, generation, PrecisionPhase::StrictAudit, view);
      const auto actual = owner.read_scalars(generation);
      const auto result = owner.download_potential(generation);
      require(actual.error == 0, "occupied or fallback source failed");
      require(std::abs(actual.energy - reference.energy) <= 1e-9, "XC energy changed");
      for (size_t element = 0; element < elements; ++element)
        require(std::abs(result[element] - potential[element]) <= 1e-8,
                 "XC potential changed");
    }
    witness.generation = 9;
    witness.coefficients = density.value;
    bool rejected = false;
    try {
      owner.enqueue(density.value, elements, 9, PrecisionPhase::StrictAudit, witness);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "valid-identity frame/density alias was not rejected");
    witness.coefficients = coefficients.value;
    owner.enqueue(density.value, elements, 9, PrecisionPhase::StrictAudit, witness);
    require(owner.read_scalars(9).error == 0, "alias failure poisoned the next generation");
    std::cout << "native occupied/fallback ownership and scientific parity passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
"""
