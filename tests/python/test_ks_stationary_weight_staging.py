"""Run the actual KS staging block against deterministic per-spin launch stubs."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def _stage_source() -> str:
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text(encoding="utf-8")
    begin = source.index("    bool staged_stationary_weights = false;")
    end = source.index("    KsPhysicalState physical;", begin)
    return source[begin:end]


@pytest.fixture(scope="module")
def staging_program(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> Path:
    directory = tmp_path_factory.mktemp("stationary-weights")
    source = directory / "probe.cpp"
    binary = directory / "probe"
    source.write_text(
        r"""
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

static unsigned builds = 0, sums = 0;
static unsigned expected_spins = 0;
int cudaGetLastError() { return 0; }
void check(int status) { assert(status == 0); }

// The production generated launcher writes separate [spin, AO, AO] blocks.
// Deliberately different alpha/beta entries make an omitted beta block visible.
void build(unsigned spins, int n, const std::int32_t* occupied, double* output) {
  ++builds;
  assert(spins == expected_spins);
  for (unsigned spin = 0; spin < spins; ++spin)
    for (int i = 0; i < n * n; ++i)
      output[spin * n * n + i] = occupied[spin] * (10.0 + 3.0 * spin + i);
}
void launch_build_weighted_density_kernel(
    unsigned grid, unsigned block, std::size_t shared, int stream, int batch, int n,
    const std::int32_t* occupied, const double*, const double*,
    const std::uint8_t* active, double* output) {
  assert(grid * block >= unsigned(n * n) && shared == 0 && stream == 7);
  assert(batch == 1 && *active == 1);
  build(1, n, occupied, output);
}
void launch_build_spin_weighted_density_kernel(
    unsigned grid, unsigned block, std::size_t shared, int stream, int batch, int n,
    const std::int32_t* occupied, const double*, const double*,
    const std::uint8_t* active, double* output) {
  assert(grid * block >= unsigned(2 * n * n) && shared == 0 && stream == 7);
  assert(batch == 1 && *active == 1);
  build(2, n, occupied, output);
}
void launch_sum_uhf_spin_matrices_kernel(
    unsigned grid, unsigned block, std::size_t shared, int stream, int batch, int n,
    const double* input, const std::uint8_t* active, double* output) {
  assert(grid * block >= unsigned(n * n) && shared == 0 && stream == 7);
  assert(batch == 1 && *active == 1);
  ++sums;
  for (int i = 0; i < n * n; ++i) output[i] = input[i] + input[n * n + i];
}
int main(int argc, char** argv) {
  assert(argc == 4);
  const unsigned spins = static_cast<unsigned>(std::atoi(argv[1]));
  const int beta = std::atoi(argv[2]);
  const bool device_validation = std::atoi(argv[3]) != 0;
  expected_spins = spins;
  const std::size_t n = 3, matrix = n * n, elements = spins * matrix;
  const int stream = 7;
  const std::int32_t occupied[2] = {2, beta};
  const std::uint8_t final_enabled[1] = {1};
  const double *final_coefficients = nullptr, *final_eigenvalues = nullptr;
  std::vector<double> density_storage(elements), first(elements, -99), second(elements, -88);
  for (std::size_t i = 0; i < matrix; ++i) {
    density_storage[i] = 2.0 + i;
    if (spins == 2) density_storage[matrix + i] = beta * (5.0 + i);
  }
  auto* density = density_storage.data();
  auto* tmp1 = first.data();
  auto* tmp2 = second.data();
  bool final_stationary_weights_ready = false;
  const bool compute_weighted_density = true;
  for (unsigned attempt = 0; attempt < 2; ++attempt) {
"""
        + _stage_source()
        + r"""
    if (device_validation && compute_weighted_density && !final_stationary_weights_ready) {
      stage_stationary_weights();
      staged_stationary_weights = true;
    }
    assert(staged_stationary_weights == (attempt == 0));
    final_stationary_weights_ready = true;
    for (std::size_t i = 0; i < matrix; ++i) {
      const double expected_w = 2.0 * (10.0 + i) + (spins == 2 ? beta * (13.0 + i) : 0);
      const double expected_d = 2.0 + i + (spins == 2 ? beta * (5.0 + i) : 0);
      assert(std::abs(tmp1[i] - expected_w) < 1e-12);
      assert(std::abs((spins == 1 ? density : tmp2)[i] - expected_d) < 1e-12);
    }
    assert(builds == 1);
    assert(sums == (spins == 2 ? 2U : 0U));
  }
}
""",
        encoding="utf-8",
    )
    return native_cxx.build_executable(
        [source], binary, compile_args=["-std=c++17", "-O0"], compile_timeout=60
    )


@pytest.mark.parametrize("spins,beta", [(1, 0), (2, 0), (2, 1), (2, 2)])
@pytest.mark.parametrize("device_validation", [False, True])
def test_stationary_staging_keeps_both_spin_weights_and_reuses_them(
    staging_program: Path, spins: int, beta: int, device_validation: bool
) -> None:
    completed = subprocess.run(
        [str(staging_program), str(spins), str(beta), str(int(device_validation))],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_stationary_weight_lease_is_published_only_after_validation() -> None:
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text(encoding="utf-8")
    begin = source.index("  VerifiedKsFinalState read_final(")
    end = source.index("  std::vector<double> download(", begin)
    read = source[begin:end]
    publish = "if (staged_stationary_weights) final_stationary_weights_ready = true;"
    assert read.count(publish) == 1
    assert read.index("validate_ks_final_state(") < read.index(publish)
    assert read.index(publish) < read.index("return verified;")
    assert "cudaStreamSynchronize" not in _stage_source()
