"""Execute generated range contractions on host to check shell workspace bounds.

The generic Cartesian evaluator is the retained arithmetic control. These checks
cover storage specialization and dual propagation; independent CPU ERIs and
GPU4PySCF endpoints separately qualify real-device scientific accuracy.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from generativeqc_compiler.integral.direct_cartesian_contraction_cuda import (
    emit_direct_cartesian_contraction_headers,
)
from generativeqc_compiler.integral.direct_pair_support_cuda import (
    emit_direct_pair_support_headers,
)
from generativeqc_compiler.integral.direct_recurrence_cuda import (
    emit_direct_recurrence_headers,
)

ROOT = Path(__file__).resolve().parents[2]


def test_all_shell_classes_preserve_range_values_and_dual_geometry(
    tmp_path: Path,
) -> None:
    """Exercise 55 shell classes, pure-axis bounds, double and all Dual3 seeds."""
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("host generated-kernel check requires c++ and ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    headers = {
        **emit_direct_cartesian_contraction_headers(),
        **emit_direct_pair_support_headers(),
        **emit_direct_recurrence_headers(),
    }
    for name, source in headers.items():
        (tmp_path / name).write_text(source)
    # Only scalar scientific helpers are executed. No CUDA runtime, launch,
    # device allocation or GPU emulation is part of this host arithmetic test.
    (tmp_path / "cuda_runtime.h").write_text(
        "#pragma once\n#define __device__\n#define __host__\n"
        "#define __forceinline__ inline\n#define __noinline__\n"
    )
    pairs = [(a, b) for a in range(4) for b in range(a + 1)]
    calls = [
        f"check<{a},{b},{c},{d}>();"
        for index, (a, b) in enumerate(pairs)
        for c, d in pairs[: index + 1]
    ]
    source = r"""
#include "generated_direct_shell_class.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
using namespace generativeqc::scf::cuda_execution;
using generativeqc::integrals::CoulombRange;
unsigned comparisons = 0;
double maximum_error = 0;
void compare(double actual, double expected) {
  const double error = std::abs(actual - expected);
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      error > 2e-13 * std::max(1.0, std::abs(expected))) {
    std::fprintf(stderr, "range workspace mismatch: %.17g versus %.17g\n", actual, expected);
    std::exit(1);
  }
  maximum_error = std::max(maximum_error, error);
  ++comparisons;
}
Angular component(unsigned angular, unsigned index) {
  const unsigned count = (angular + 1) * (angular + 2) / 2;
  index %= count;
  for (int x = angular; x >= 0; --x)
    for (int y = angular - x; y >= 0; --y)
      if (index-- == 0) return {static_cast<unsigned>(x), static_cast<unsigned>(y),
                               angular - static_cast<unsigned>(x + y)};
  std::abort();
}
template<unsigned A, unsigned B, unsigned C, unsigned D>
void check() {
  const unsigned momenta[4]{A,B,C,D};
  const double exponents[4]{0.45,0.8,1.2,0.7};
  for (unsigned sample = 0; sample < 9; ++sample) {
    Angular angular[4];
    for (unsigned i = 0; i < 4; ++i) {
      // The pure-axis tuples reach the maximum pair power, including ffff t=6.
      angular[i] = sample < 6 ? component(momenta[i], sample * (2*i+1) + i)
          : Angular{sample == 6 ? momenta[i] : 0, sample == 7 ? momenta[i] : 0,
                    sample == 8 ? momenta[i] : 0};
    }
    for (unsigned seed = 0; seed < 4; ++seed) {
      Vec3<Dual3> centers[4];
      for (unsigned i = 0; i < 4; ++i) {
        const double shift = 0.13 * sample;
        centers[i] = {{0.1 + 0.23*i + shift, seed == i ? 1.0 : 0.0,0,0},
                      {-0.3 + 0.11*i*i,0,seed == i ? 1.0 : 0.0,0},
                      {0.7 - 0.19*i - shift*i,0,0,seed == i ? 1.0 : 0.0}};
      }
      for (const auto range : {CoulombRange::Full, CoulombRange::Short, CoulombRange::Long}) {
        const double omega = range == CoulombRange::Full ? 0.0 : (sample % 2 ? 0.3 : 2.1);
        const auto actual = primitive_eri_cartesian_shell_pairs<A,B,C,D>(
            exponents[0],centers[0],angular[0], exponents[1],centers[1],angular[1],
            exponents[2],centers[2],angular[2], exponents[3],centers[3],angular[3],range,omega);
        const auto expected = primitive_eri_cartesian<A+B+C+D>(
            exponents[0],centers[0],angular[0], exponents[1],centers[1],angular[1],
            exponents[2],centers[2],angular[2], exponents[3],centers[3],angular[3],range,omega);
        compare(actual.value,expected.value);
        compare(actual.derivative_x,expected.derivative_x);
        compare(actual.derivative_y,expected.derivative_y);
        compare(actual.derivative_z,expected.derivative_z);
        if (seed == 0) {
          // Scalar values instantiate a different radial moment count from AD.
          Vec3<double> positions[4];
          for (unsigned i = 0; i < 4; ++i)
            positions[i] = {centers[i].x.value, centers[i].y.value, centers[i].z.value};
          const double value = primitive_eri_cartesian_shell_pairs<A,B,C,D>(
              exponents[0],positions[0],angular[0], exponents[1],positions[1],angular[1],
              exponents[2],positions[2],angular[2], exponents[3],positions[3],angular[3],range,omega);
          const double control = primitive_eri_cartesian<A+B+C+D>(
              exponents[0],positions[0],angular[0], exponents[1],positions[1],angular[1],
              exponents[2],positions[2],angular[2], exponents[3],positions[3],angular[3],range,omega);
          compare(value,control);
        }
      }
    }
  }
}
int main() {
CALLS
  std::printf("%u value/dual comparisons; maximum error %.17g\n", comparisons, maximum_error);
}
""".replace("CALLS", "\n".join(calls))
    path = tmp_path / "probe.cpp"
    path.write_text(source)
    executable = tmp_path / "probe"
    environment = {**os.environ, "CCACHE_BASEDIR": str(ROOT)}
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            f"-I{tmp_path}",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
            str(path),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
    )
    result = subprocess.run(
        [str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "25245 value/dual comparisons" in result.stdout
