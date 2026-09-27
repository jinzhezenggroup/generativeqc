"""Host-execute the production collection/seed kernels' error contract.

CUDA lane indices and atomics are scalar fixtures here. This is not a CUDA
scheduling or molecular-force qualification.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _function(source: str, name: str) -> str:
    start = source.index(f"__global__ void {name}(")
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end].replace("__global__ ", "", 1)


PREFIX = r"""
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
using std::isfinite;
using std::signbit;
struct Index { std::size_t x{}; } blockIdx, threadIdx, blockDim{1};
void atomicExch(int* out, int value) { *out = value; }
double __longlong_as_double(unsigned long long value) {
  return std::bit_cast<double>(static_cast<std::uint64_t>(value));
}
namespace vibeqc::dft {
struct GridTaskView {
  std::size_t npoint;
  const double* features;
  const int* error;
};
}
"""

MAIN = r"""
int main(int argc, char** argv) {
  if (argc != 2) return 1;
  const int mode = std::atoi(argv[1]);
  constexpr std::size_t n = 4, offset = 1, count = 2;
  std::array<double, 13 * count> features{};
  for (std::size_t i = 0; i < count; ++i) {
    features[i] = 0.7;
    features[5 * count + i] = 0.4;
    features[count + i] = 0.1;
    features[6 * count + i] = 0.2;
  }
  int producer_error = mode == 1 ? 7 : (mode == 2 ? -7 : 0);
  int collect_error = 0, domain_error = 0, pair_error = 0;
  std::array<double, n> density;
  std::array<double, 3 * n> gradient;
  density.fill(91.0);
  gradient.fill(92.0);
  if (mode == 3) features[0] = std::numeric_limits<double>::quiet_NaN();
  vibeqc::dft::GridTaskView view{count, features.data(), mode == 4 ? nullptr : &producer_error};
  const auto begin = mode == 6 ? n : offset;
  for (std::size_t lane = 0; lane < count; ++lane) {
    threadIdx.x = lane;
    collect_total_features_kernel(view, begin, n, density.data(), gradient.data(), &collect_error);
  }
  if (density.front() != 91.0 || density.back() != 91.0) return 2;
  if (gradient.front() != 92.0 || gradient.back() != 92.0) return 3;
  const bool expected_failure = mode == 1 || mode == 2 || mode == 3 || mode == 6;
  if ((collect_error != 0) != expected_failure) return 4;
  if (mode == 1 || mode == 2) {
    for (std::size_t i = offset; i < offset + count; ++i) {
      if (density[i] != 0.0) return 5;
      for (std::size_t k = 0; k < 3; ++k)
        if (gradient[3 * i + k] != 0.0) return 6;
    }
  } else if (!expected_failure) {
    for (std::size_t i = offset; i < offset + count; ++i) {
      if (std::abs(density[i] - 1.1) > 1e-15) return 7;
      if (std::abs(gradient[3 * i] - 0.3) > 1e-15) return 8;
    }
  }
  // A later successful tile must not clear an earlier collection failure.
  if (mode == 1 || mode == 2) {
    producer_error = 0;
    threadIdx.x = 0;
    collect_total_features_kernel(view, offset, n, density.data(), gradient.data(), &collect_error);
    if (!collect_error) return 9;
  }
  std::array<double, n> weights;
  std::array<double, 3 * n> point_derivative;
  std::array<double, 6 * n> seeds;
  weights.fill(mode == 5 ? 0.0 : (mode == 7 ? -0.0 : 1.0));
  point_derivative.fill(3.0);
  seeds.fill(2.0);
  for (std::size_t lane = 0; lane < n; ++lane) {
    threadIdx.x = lane;
    pack_force_seeds_kernel(n, weights.data(), point_derivative.data(), seeds.data(),
                           &collect_error, &domain_error, &pair_error);
  }
  for (std::size_t row = 0; row < 6; ++row)
    for (std::size_t i = 0; i < n; ++i) {
      const double value = seeds[row * n + i];
      if (expected_failure) {
        if (!std::isnan(value)) return 10;
      } else if (mode == 5) {
        const double expected = row >= 2 && row <= 4 ? 3.0 : 2.0;
        if (value != expected) return 11;
      } else if (mode == 7) {
        if (value != 0.0) return 12;
      } else if (value != (row >= 2 && row <= 4 ? 3.0 : 2.0)) return 13;
    }
}
"""


@pytest.fixture(scope="module")
def error_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/dft/nonlocal_correlation/vv10_runtime_cuda.cu").read_text()
    directory = tmp_path_factory.mktemp("resident-nonlocal-error")
    path, executable = directory / "probe.cpp", directory / "probe"
    path.write_text(
        "\n".join(
            (
                PREFIX,
                _function(source, "collect_total_features_kernel"),
                _function(source, "pack_force_seeds_kernel"),
                MAIN,
            )
        )
    )
    compiled = subprocess.run(
        [compiler, "-std=c++20", "-O0", str(path), "-o", str(executable)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return executable


@pytest.mark.parametrize("mode", range(8))
def test_resident_collection_preserves_producer_error(
    error_probe: Path, mode: int
) -> None:
    result = subprocess.run(
        [str(error_probe), str(mode)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, (
        mode,
        result.returncode,
        result.stdout,
        result.stderr,
    )


def test_zero_weight_row_mask_distinguishes_active_from_density_inactive() -> None:
    source = (ROOT / "src/dft/nonlocal_correlation/vv10_runtime_cuda.cu").read_text()
    assert "effective_weights[i] = inactive ? -0.0" in source
    assert "if (signbit(weighted_i))" in source
    assert "if (signbit(effective_weights[i]))" in source


def _two_point_vv10_energy(weight0: float) -> float:
    b = 6.0
    rho = (0.8, 1.1)
    weights = (weight0, 1.0)
    positions = (0.0, 1.0)
    beta = (3.0 / (b * b)) ** 0.75 / 32.0
    omega = tuple(math.sqrt((4.0 * math.pi / 3.0) * value) for value in rho)
    kappa = tuple(
        b * 1.5 * math.pi * (value / (9.0 * math.pi)) ** (1.0 / 6.0) for value in rho
    )
    total = 0.0
    for i in range(2):
        pair_sum = 0.0
        for j in range(2):
            r2 = (positions[j] - positions[i]) ** 2
            gi = omega[i] * r2 + kappa[i]
            gj = omega[j] * r2 + kappa[j]
            phi = -1.5 / (gi * gj * (gi + gj))
            pair_sum += weights[j] * rho[j] * phi
        total += weights[i] * rho[i] * (beta + 0.5 * pair_sum)
    return total


def test_density_active_zero_weight_has_nonzero_weight_derivative() -> None:
    b, rho0, rho1 = 6.0, 0.8, 1.1
    beta = (3.0 / (b * b)) ** 0.75 / 32.0
    omega0 = math.sqrt((4.0 * math.pi / 3.0) * rho0)
    omega1 = math.sqrt((4.0 * math.pi / 3.0) * rho1)
    kappa0 = b * 1.5 * math.pi * (rho0 / (9.0 * math.pi)) ** (1.0 / 6.0)
    kappa1 = b * 1.5 * math.pi * (rho1 / (9.0 * math.pi)) ** (1.0 / 6.0)
    phi01 = -1.5 / (
        (omega0 + kappa0) * (omega1 + kappa1) * (omega0 + kappa0 + omega1 + kappa1)
    )
    expected = rho0 * (beta + rho1 * phi01)
    assert expected != 0.0
    for step in (1.0e-4, 1.0e-5, 1.0e-6):
        finite_difference = (
            _two_point_vv10_energy(step) - _two_point_vv10_energy(-step)
        ) / (2.0 * step)
        assert finite_difference == pytest.approx(expected, rel=2.0e-8, abs=2.0e-11)
