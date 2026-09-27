"""Host-execute generic and exact-class Direct force coefficient boundaries."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from vibeqc_compiler.integral.lowering.dispatch import emit_shell_class_fused_cuda
from vibeqc_compiler.integral.lowering.fock_accumulation import (
    emit_direct_force_density_coefficient,
)
from vibeqc_compiler.integral.shell_spec import DPPP_SPEC


def _function(source: str, name: str) -> str:
    marker = f"double {name}("
    position = source.index(marker)
    start = source.rfind("template <bool Unrestricted>", 0, position)
    opening = source.index("{", position)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end].replace("__device__ ", "").replace("__forceinline__ ", "")


@pytest.fixture(scope="module")
def coefficient_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")

    generic = _function(
        emit_direct_force_density_coefficient(),
        "direct_force_density_coefficient_scaled",
    )
    exact = _function(
        emit_shell_class_fused_cuda(DPPP_SPEC),
        "generated_dppp_density_coefficient_scaled",
    )
    source = (
        r"""
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>

inline std::size_t matrix_index(
    std::size_t i, std::size_t j, std::size_t n) {
  return i * n + j;
}
inline std::size_t generated_dppp_matrix_index(
    std::size_t i, std::size_t j, std::size_t n) {
  return i * n + j;
}
inline bool unique_eri_symmetry_permutation(
    unsigned permutation, std::size_t, std::size_t, std::size_t, std::size_t) {
  return permutation < 8;
}
inline void eri_symmetry_permutation(
    unsigned permutation, std::size_t i, std::size_t j, std::size_t k, std::size_t l,
    std::size_t& a, std::size_t& b, std::size_t& c, std::size_t& d) {
  const std::size_t values[8][4] = {
      {i,j,k,l}, {j,i,k,l}, {i,j,l,k}, {j,i,l,k},
      {k,l,i,j}, {l,k,i,j}, {k,l,j,i}, {l,k,j,i}};
  a = values[permutation][0];
  b = values[permutation][1];
  c = values[permutation][2];
  d = values[permutation][3];
}
struct GeneratedDpppShellTask {
  std::size_t matrix_order;
  std::size_t spin_offset;
  std::size_t density_offset;
};
"""
        + generic
        + "\n"
        + exact
        + r"""

void set_symmetric(std::array<double, 16>& d, std::size_t i, std::size_t j, double value) {
  d[i * 4 + j] = value;
  d[j * 4 + i] = value;
}
bool close(double a, double b) {
  return std::isfinite(a) && std::isfinite(b) &&
         std::abs(a - b) <= 1e-12 * std::max({1.0, std::abs(a), std::abs(b)});
}
int main() {
  GeneratedDpppShellTask task{4, 0, 0};

  // K=0: unused exchange products overflow, while requested J is finite zero.
  std::array<double, 16> d{};
  set_symmetric(d, 0, 1, 1e200);
  set_symmetric(d, 0, 2, 1e200);
  set_symmetric(d, 1, 3, 1e200);
  const double generic_j = direct_force_density_coefficient_scaled<false>(
      4, 0, 0, d.data(), 0, 1, 2, 3, 1.0, 0.0);
  const double exact_j = generated_dppp_density_coefficient_scaled<false>(
      task, 0, 1, 2, 3, d.data(), 1.0, 0.0);
  if (!close(generic_j, exact_j)) return 1;

  // J=0: unused Coulomb product overflows, while requested exchange is finite.
  d.fill(0.0);
  set_symmetric(d, 0, 1, 1e200);
  set_symmetric(d, 2, 3, 1e200);
  set_symmetric(d, 0, 2, 1.0);
  set_symmetric(d, 1, 3, 1.0);
  const double generic_k = direct_force_density_coefficient_scaled<false>(
      4, 0, 0, d.data(), 0, 1, 2, 3, 0.0, 1.0);
  const double exact_k = generated_dppp_density_coefficient_scaled<false>(
      task, 0, 1, 2, 3, d.data(), 0.0, 1.0);
  if (!close(generic_k, exact_k)) return 2;

  d.fill(1e200);
  const double generic_zero = direct_force_density_coefficient_scaled<false>(
      4, 0, 0, d.data(), 0, 1, 2, 3, 0.0, 0.0);
  const double exact_zero = generated_dppp_density_coefficient_scaled<false>(
      task, 0, 1, 2, 3, d.data(), 0.0, 0.0);
  if (generic_zero != 0.0 || exact_zero != 0.0) return 3;
}
"""
    )
    directory = tmp_path_factory.mktemp("direct-force-coefficients")
    source_path = directory / "probe.cpp"
    executable = directory / "probe"
    source_path.write_text(source)
    compiled = subprocess.run(
        [compiler, "-std=c++20", "-O0", str(source_path), "-o", str(executable)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return executable


def test_exact_class_matches_generic_when_inactive_term_would_overflow(
    coefficient_probe: Path,
) -> None:
    result = subprocess.run(
        [str(coefficient_probe)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
