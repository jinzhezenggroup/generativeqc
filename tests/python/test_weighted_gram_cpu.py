"""Independent CPU weighted-Gram arithmetic and primitive-provider contract."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_cpu_weighted_gram_math_storage_and_failure_contract(tmp_path: Path) -> None:
    compiler, ccache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or ccache is None:
        pytest.skip("host C++ compiler and ccache required")
    subprocess.run([ccache, "--version"], check=True, capture_output=True, timeout=15)
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            sys.executable,
            "-S",
            str(root / "tools/generate_weighted_gram_native.py"),
            "--output",
            str(tmp_path / "generated_weighted_gram_native.hpp"),
        ],
        check=True,
        capture_output=True,
        timeout=45,
    )
    source = tmp_path / "weighted_gram.cpp"
    source.write_text(r"""
#include "tensor/weighted_gram.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>

namespace gram = generativeqc::tensor::weighted_gram;
constexpr double sentinel = 123.25;
static const double* expected_coefficients;
static double *expected_panel, *expected_output;
static std::int32_t expected_n;
static unsigned calls;

static void primitive_gemm(int layout, int left_transpose, int right_transpose,
                           std::int32_t m, std::int32_t n, std::int32_t k,
                           double alpha, const double* a, std::int32_t lda,
                           const double* b, std::int32_t ldb, double beta,
                           double* c, std::int32_t ldc) {
  if (layout != 102 || left_transpose != 111 || right_transpose != 112 ||
      m != expected_n || n != expected_n || k != expected_n ||
      lda != expected_n || ldb != expected_n || ldc != expected_n ||
      alpha != 1.0 || beta != 0.0 || a != expected_panel ||
      b != expected_coefficients || c != expected_output) std::abort();
  ++calls;
  // A provider substitute, deliberately independent of generated scalar helpers.
  // Match DGEMM's beta==0 rule: do not read the prior output, which may be NaN.
  for (std::int32_t column = 0; column < n; ++column) {
    for (std::int32_t row = 0; row < m; ++row) {
      double sum = 0.0;
      for (std::int32_t orbital = 0; orbital < k; ++orbital)
        sum += a[row + orbital * lda] * b[column + orbital * ldb];
      c[row + column * ldc] = sum;
    }
  }
}

int main() {
  static_assert(sizeof(std::int32_t) == 4);
  static_assert(std::is_same_v<gram::CblasDgemmLp64, decltype(&primitive_gemm)>);
  for (std::int32_t n : {1, 2, 3, 5, 9}) {
    const std::size_t count = static_cast<std::size_t>(n) * n;
    std::array<double, 83> coefficients, panel, output;
    std::array<double, 11> weights, eigenvalues, energy_weights;
    for (auto* matrix : {&coefficients, &panel, &output}) matrix->fill(sentinel);
    for (auto* vector : {&weights, &eigenvalues, &energy_weights}) vector->fill(sentinel);
    double *c = coefficients.data() + 1, *p = panel.data() + 1, *out = output.data() + 1;
    double *w = weights.data() + 1, *e = eigenvalues.data() + 1;
    double* ew = energy_weights.data() + 1;
    for (std::int32_t orbital = 0; orbital < n; ++orbital) {
      w[orbital] = (orbital + 1.0) / (n + 1.0); // fractional, nonuniform occupations
      e[orbital] = orbital % 2 == 0 ? -1.75 - orbital : 0.625 + orbital;
      ew[orbital] = w[orbital];
      for (std::int32_t row = 0; row < n; ++row)
        c[row + orbital * n] = (row % 2 ? -1.0 : 1.0) *
                               (0.125 + (row + 1.0) / (orbital + 2.0));
    }
    const auto original_coefficients = coefficients;
    const auto original_weights = weights;
    const auto original_eigenvalues = eigenvalues;
    if (!gram::energy_weights_inplace(n, e, ew)) return 1;
    for (std::int32_t orbital = 0; orbital < n; ++orbital)
      if (ew[orbital] != w[orbital] * e[orbital]) return 2;
    for (const double* scale : {w, ew}) {
      expected_n = n;
      expected_coefficients = c;
      expected_panel = p;
      expected_output = out;
      calls = 0;
      std::fill_n(out, count, std::numeric_limits<double>::quiet_NaN());
      if (!gram::execute_column_major(primitive_gemm, n, c, scale, p, out) || calls != 1)
        return 3;
      for (std::int32_t column = 0; column < n; ++column) {
        for (std::int32_t row = 0; row < n; ++row) {
          // The mathematical oracle uses wider arithmetic and no shared code.
          long double expected = 0.0L, magnitude = 0.0L;
          for (std::int32_t orbital = 0; orbital < n; ++orbital) {
            const long double term = static_cast<long double>(c[row + orbital * n]) *
                                     scale[orbital] * c[column + orbital * n];
            expected += term;
            magnitude += std::abs(term);
          }
          if (std::abs(static_cast<long double>(out[row + column * n]) - expected) >
              8 * n * std::numeric_limits<double>::epsilon() * std::max(1.0L, magnitude))
            return 4;
          if (p[row + column * n] != c[row + column * n] * scale[column]) return 5;
        }
      }
      for (auto* matrix : {&coefficients, &panel, &output}) {
        if ((*matrix)[0] != sentinel) return 6;
        for (std::size_t i = count + 1; i < matrix->size(); ++i)
          if ((*matrix)[i] != sentinel) return 7;
      }
    }
    if (coefficients != original_coefficients || weights != original_weights ||
        eigenvalues != original_eigenvalues || energy_weights[0] != sentinel) return 8;
    for (std::size_t i = n + 1; i < energy_weights.size(); ++i)
      if (energy_weights[i] != sentinel) return 9;
  }

  // A late scaling failure publishes only its checked, orbital-major prefix;
  // neither the failing element nor any output element may be modified.
  std::array<double, 9> coefficients{1., 2., 3., 4., 5., 1e308, 7., 8., 9.};
  std::array<double, 3> weights{0.5, 2., -1.};
  std::array<double, 9> panel, output;
  panel.fill(sentinel);
  output.fill(sentinel);
  calls = 0;
  if (gram::execute_column_major(primitive_gemm, 3, coefficients.data(), weights.data(),
                                panel.data(), output.data()) || calls != 0) return 10;
  for (std::size_t i = 0; i < panel.size(); ++i) {
    const double expected = i < 5 ? coefficients[i] * weights[i / 3] : sentinel;
    if (panel[i] != expected || output[i] != sentinel) return 11;
  }
  // In-place transform has the same prefix-publication failure contract.
  std::array<double, 3> energies{-2., 1e308, 4.}, in_place{0.5, 2., 0.25};
  if (gram::energy_weights_inplace(3, energies.data(), in_place.data()) ||
      in_place != std::array<double, 3>{-1., 2., 0.25}) return 12;
  // Signed zeros and generated FMA updates retain the original scalar contract.
  double out = sentinel;
  if (!gram::generated::energy_weight(0., -2., out) || !std::signbit(out)) return 13;
  if (!gram::generated::density_update(1e308, 2., -1e308, out) ||
      out != std::fma(1e308, 2., -1e308)) return 14;
}
""")
    obj = tmp_path / "weighted_gram.o"
    subprocess.run(
        [
            ccache,
            compiler,
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            "-I",
            str(root / "src"),
            "-I",
            str(tmp_path),
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "CCACHE_BASEDIR": str(root)},
        timeout=60,
    )
    binary = tmp_path / "weighted_gram"
    subprocess.run(
        [compiler, str(obj), "-o", str(binary)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    result = subprocess.run(
        [str(binary)], check=False, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
