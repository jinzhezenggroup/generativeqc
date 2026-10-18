"""Independent displaced-integral checks for fused relative-center responses."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from generativeqc_compiler.integral.direct_cartesian_contraction_cuda import (
    emit_direct_cartesian_contraction_headers,
)
from generativeqc_compiler.integral.direct_order2_shell_cuda import (
    emit_direct_order2_shell_header,
)
from generativeqc_compiler.integral.direct_pair_support_cuda import (
    emit_direct_pair_support_headers,
)
from generativeqc_compiler.integral.direct_recurrence_cuda import (
    emit_direct_recurrence_headers,
)
from generativeqc_compiler.integral.direct_relative_response_cuda import (
    emit_direct_relative_response_support,
)
from generativeqc_compiler.integral.direct_source_contraction_cuda import (
    emit_direct_source_contraction_header,
)

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_relative_responses_match_displaced_integrals(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Check every center, atom coalescing, zero displacements and decay underflow."""
    headers = {
        **emit_direct_cartesian_contraction_headers(),
        **emit_direct_pair_support_headers(),
        **emit_direct_recurrence_headers(),
        "generated_direct_order2_shell.cuh": emit_direct_order2_shell_header(),
        "source.cuh": emit_direct_source_contraction_header(),
        "relative.cuh": '#include "source.cuh"\n'
        "namespace generativeqc::scf::cuda_execution {\n"
        + emit_direct_relative_response_support()
        + "\n}\n",
    }
    for name, text in headers.items():
        (tmp_path / name).write_text(text)
    (tmp_path / "cuda_runtime.h").write_text(
        "#pragma once\n#include <cmath>\n#include <algorithm>\n"
        "#define __device__\n#define __host__\n#define __forceinline__ inline\n"
        "#define __noinline__\nusing std::min; using std::max;\n"
        "inline unsigned __popc(unsigned value) { return __builtin_popcount(value); }\n"
    )
    source = tmp_path / "probe.cpp"
    source.write_text(PROBE)
    executable = tmp_path / "probe"
    native_cxx.build_executable(
        [source],
        executable,
        compile_args=[
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            f"-I{tmp_path}",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ],
        compile_timeout=180,
        link_timeout=120,
    )
    result = subprocess.run(
        [str(executable)], check=False, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "relative responses match independent displacements PASS" in result.stdout
    print(result.stdout)


PROBE = r"""
#include "relative.cuh"
#include "generated_direct_shell_class.cuh"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
using namespace generativeqc::scf::cuda_execution;
unsigned checks = 0;
double maximum_error = 0.0;
double maximum_ad_error = 0.0;

double normalization(double exponent, const Angular& angular) {
  double result = pow(2 * exponent / kPi, 0.75);
  for (unsigned axis = 0; axis < 3; ++axis)
    for (unsigned degree = 1; degree <= angular_axis(angular, axis); ++degree)
      result *= sqrt(4 * exponent / (2 * degree - 1));
  return result;
}

template <unsigned First, unsigned Second, unsigned Third, unsigned Fourth>
void qualify() {
  constexpr unsigned orders[4]{First, Second, Third, Fourth};
  for (double scale : {0.03, 1.0, 30.0}) {
    const double exponents[4]{0.8 * scale, 0.6 * scale, 0.7 * scale, 0.9 * scale};
    for (auto atoms : {std::array<unsigned, 4>{0,1,2,3}, {0,0,1,2}, {0,1,0,2},
                      {0,1,0,1}, {0,0,0,1}, {0,0,0,0}})
      for (unsigned geometry = 0; geometry < 3; ++geometry)
        for (unsigned orientation = 0; orientation < 9; ++orientation) {
          std::array<Vec3<double>, 4> positions{{{0.1,-0.2,-0.8}, {0.3,0.1,0.7},
                                               {-0.5,0.6,0.2}, {0.8,-0.4,0.3}}};
          if (geometry == 1) positions.fill(positions[0]);
          if (geometry == 2) positions[1].x = 1e4;
          std::array<Vec3<double>, 4> centers;
          Angular angular[4]{};
          for (unsigned center = 0; center < 4; ++center) {
            centers[center] = positions[atoms[center]];
            for (unsigned quantum = 0; quantum < orders[center]; ++quantum)
              add_angular_axis(angular[center],
                  (center + quantum + orientation * (quantum + 1)) % 3, 1);
          }
          double coefficient = 1.0;
          for (unsigned center = 0; center < 4; ++center)
            coefficient *= normalization(exponents[center], angular[center]);
          DirectRelativePairDerivativeRecurrence recurrence;
          recurrence.p = exponents[0] + exponents[1];
          recurrence.q = exponents[2] + exponents[3];
          for (unsigned pair = 0; pair < 2; ++pair) {
            const unsigned left = 2 * pair, right = left + 1;
            PrimitivePairData cached{};
            cached.exponent_sum = exponents[left] + exponents[right];
            cached.reduced_exponent = exponents[left] * exponents[right] / cached.exponent_sum;
            cached.product_center = product_center(exponents[left], centers[left],
                                                   exponents[right], centers[right]);
            for (unsigned axis = 0; axis < 3; ++axis)
              prepare_direct_relative_pair(cached, exponents[left], exponents[right],
                  centers[left], centers[right], pair ? recurrence.ket : recurrence.bra,
                  orders[left], orders[right], axis);
          }
          const auto product_p = product_center(exponents[0], centers[0], exponents[1], centers[1]);
          const auto product_q = product_center(exponents[2], centers[2], exponents[3], centers[3]);
          fill_coulomb<9>(recurrence.p * recurrence.q / (recurrence.p + recurrence.q),
                          product_p, product_q, recurrence.coulomb);
          double responses[3][3]{};
          contract_direct_relative_response_basis(angular[0], angular[1], angular[2], angular[3],
                                               recurrence, responses);
          const double prefactor = coefficient * 2.0 * pow(kPi, 2.5) /
              (recurrence.p * recurrence.q * sqrt(recurrence.p + recurrence.q));
          for (unsigned atom = 0; atom < 4; ++atom)
            for (unsigned axis = 0; axis < 3; ++axis) {
              Vec3<Dual3> seeded[4];
              for (unsigned center = 0; center < 4; ++center) {
                const double seed = double(atoms[center] == atom);
                seeded[center] = {{centers[center].x, seed, 0.0, 0.0},
                    {centers[center].y, 0.0, seed, 0.0},
                    {centers[center].z, 0.0, 0.0, seed}};
              }
              const auto retained = primitive_eri_cartesian_shell_pairs<First,Second,Third,Fourth,Dual3>(
                  exponents[0],seeded[0],angular[0],exponents[1],seeded[1],angular[1],
                  exponents[2],seeded[2],angular[2],exponents[3],seeded[3],angular[3]);
              const double retained_derivatives[3]{coefficient * retained.derivative_x,
                  coefficient * retained.derivative_y, coefficient * retained.derivative_z};
              const double bra = double(atoms[0] == atom) - double(atoms[1] == atom);
              const double ket = double(atoms[2] == atom) - double(atoms[3] == atom);
              const double product = (exponents[0] * double(atoms[0] == atom) +
                  exponents[1] * double(atoms[1] == atom)) / recurrence.p -
                  (exponents[2] * double(atoms[2] == atom) +
                   exponents[3] * double(atoms[3] == atom)) / recurrence.q;
              const double actual = prefactor * (bra * responses[0][axis] +
                  ket * responses[1][axis] + product * responses[2][axis]);
              if (std::abs(actual - retained_derivatives[axis]) >
                  2e-10 * (1.0 + std::abs(retained_derivatives[axis]))) std::exit(2);
              maximum_ad_error = std::max(maximum_ad_error,
                  std::abs(actual - retained_derivatives[axis]));
              auto value = [&](double displacement) {
                auto moved = centers;
                for (unsigned center = 0; center < 4; ++center)
                  if (atoms[center] == atom) {
                    double* coordinate = axis == 0 ? &moved[center].x :
                                         axis == 1 ? &moved[center].y : &moved[center].z;
                    *coordinate += displacement;
                  }
                const auto origin = moved[0];
                for (auto& position : moved) {
                  position.x -= origin.x;
                  position.y -= origin.y;
                  position.z -= origin.z;
                }
                return coefficient * primitive_eri_cartesian_shell_pairs<First,Second,Third,Fourth,double>(
                    exponents[0],moved[0],angular[0],exponents[1],moved[1],angular[1],
                    exponents[2],moved[2],angular[2],exponents[3],moved[3],angular[3]);
              };
              const double step = 1e-4 / std::sqrt(scale);
              const double samples[4]{value(-2 * step), value(-step), value(step), value(2 * step)};
              const double expected = (samples[0] - 8 * samples[1] +
                  8 * samples[2] - samples[3]) / (12 * step);
              const double error = std::abs(actual - expected) / (1.0 + std::abs(expected));
              const double cancellation_bound = 64 * std::numeric_limits<double>::epsilon() *
                  std::max({std::abs(samples[0]), std::abs(samples[1]),
                            std::abs(samples[2]), std::abs(samples[3])}) / step;
              if (!std::isfinite(actual) || std::abs(actual - expected) >
                  3e-9 * (1.0 + std::abs(expected)) + cancellation_bound) {
                std::fprintf(stderr, "relative response failed order=%u%u%u%u scale=%g geometry=%u "
                    "orientation=%u atoms=%u%u%u%u atom=%u axis=%u actual=%g expected=%g error=%g value=%g\n",
                    First, Second, Third, Fourth, scale, geometry, orientation,
                    atoms[0], atoms[1], atoms[2], atoms[3], atom, axis, actual, expected, error, samples[0]);
                std::exit(1);
              }
              maximum_error = std::max(maximum_error, error);
              ++checks;
            }
        }
  }
}

int main() {
  qualify<2,2,2,0>();
  qualify<2,2,1,1>();
  qualify<2,1,2,1>();
  qualify<2,2,2,1>();
  qualify<2,2,2,2>();
  std::printf("relative responses match independent displacements PASS checks=%u FD=%g AD=%g\n",
              checks, maximum_error, maximum_ad_error);
}
"""
