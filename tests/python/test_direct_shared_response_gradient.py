"""Independent displaced-value and retained-AD gates for shared scalar responses."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from generativeqc_compiler.integral.direct_cartesian_contraction_cuda import (
    emit_direct_cartesian_contraction_headers,
)
from generativeqc_compiler.integral.direct_order2_shell_cuda import (
    emit_direct_order2_shell_header,
)
from generativeqc_compiler.integral.direct_pair_materialized_gradient_cuda import (
    emit_direct_pair_materialized_gradient_support,
)
from generativeqc_compiler.integral.direct_pair_support_cuda import (
    emit_direct_pair_support_headers,
)
from generativeqc_compiler.integral.direct_recurrence_cuda import (
    emit_direct_recurrence_headers,
)
from generativeqc_compiler.integral.direct_source_contraction_cuda import (
    emit_direct_source_contraction_header,
)

ROOT = Path(__file__).resolve().parents[2]


def test_shared_response_keeps_bounded_default_and_explicit_fallbacks() -> None:
    """The L+1 scalar owner cannot replace unsupported or optional AD algebra."""
    source = emit_direct_source_contraction_header()
    assert "struct DirectCoulombSpatialResponse" in source
    assert (
        "using MaterializedCoulombSpatialResponse = DirectCoulombSpatialResponse<9>"
        in source
    )
    assert "CoulombAuxiliary<double, AngularOrder + 1U> coulomb" in source
    assert "if (centers <= 1U) return result" in source
    dispatch = (ROOT / "src/scf/cuda/direct_force_quartet.cuh").read_text()
    assert (
        dispatch.count("contracted_direct_shared_response_gradient<AngularOrder>") == 2
    )
    assert dispatch.count("AngularOrder < kMaximumCoulombOrder") == 2
    assert dispatch.count("(batch.direct_coulomb_reachable & 2U) == 0U") == 2
    assert dispatch.count("(batch.direct_hermite_convolution & 2U) == 0U") == 2
    assert "range == generativeqc::integrals::CoulombRange::Long &&" in dispatch


def test_shared_atomic_response_matches_displaced_values(tmp_path: Path) -> None:
    """Check signed contractions, repeated atoms, range limits and exact work."""
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("host gradient probes require c++ and ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    headers = {
        **emit_direct_pair_support_headers(),
        **emit_direct_recurrence_headers(),
        **emit_direct_cartesian_contraction_headers(),
        "generated_direct_order2_shell.cuh": emit_direct_order2_shell_header(),
        "source.cuh": emit_direct_source_contraction_header(),
    }
    support = emit_direct_pair_materialized_gradient_support()
    begin = support.index(
        "template <unsigned MaximumAngular>\nstruct DirectCoulombSpatialResponse"
    )
    end = support.index("/** Bra Hermite responses", begin)
    headers["response.cuh"] = SUPPORT_PREFIX + support[begin:end]
    for name, source in headers.items():
        (tmp_path / name).write_text(source)
    (tmp_path / "cuda_runtime.h").write_text(
        "#pragma once\n#include <cmath>\n#include <algorithm>\n"
        "#define __device__\n#define __host__\n#define __forceinline__ inline\n"
        "#define __noinline__\nusing std::min; using std::max;\n"
        "inline unsigned __popc(unsigned value) { return __builtin_popcount(value); }\n"
    )
    source = tmp_path / "probe.cpp"
    source.write_text(PROBE)
    executable = tmp_path / "probe"
    build = subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            f"-I{tmp_path}",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
            str(source),
            "-o",
            str(executable),
        ],
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=180, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "shared atomic response and exact work PASS" in result.stdout
    print(result.stdout)


SUPPORT_PREFIX = r"""
#include "source.cuh"
#include <limits>
namespace generativeqc::scf::cuda_execution {
struct MaterializedDirectPairWork {
  unsigned long long bra_preparations{}, ket_preparations{}, coulomb_preparations{};
  unsigned long long component_contractions{}, published_components{};
};
inline void atomicAdd(unsigned long long* destination, unsigned long long value) {
  *destination += value;
}
"""


PROBE = r"""
#include "response.cuh"
}
#include <array>
#include <cstdio>
#include <cstdlib>
using namespace generativeqc::scf::cuda_execution;
using generativeqc::integrals::CoulombRange;
unsigned checks = 0;
double maximum_ad_error = 0.0, maximum_displaced_error = 0.0;

template <unsigned First, unsigned Second, unsigned Third, unsigned Fourth>
void qualify() {
  constexpr unsigned order = First + Second + Third + Fourth;
  const std::uint8_t shell_angular[4]{First, Second, Third, Fourth};
  const std::int32_t ao_shells[4]{0, 1, 2, 3};
  const std::int64_t primitive_offsets[5]{0, 2, 4, 6, 8};
  const double base_exponents[8]{0.8, 0.3, 0.6, 1.1, 0.7, 0.4, 0.9, 0.5};
  const double coefficients[8]{0.7, -0.31, 0.7, -0.31, 0.7, -0.31, 0.7, -0.31};
  const double ao_coefficients[4]{0.7, 0.9, -0.8, 1.1};
  for (double scale : {1e-3, 1.0, 1e3}) {
  double exponents[8]{};
  for (unsigned primitive = 0; primitive < 8U; ++primitive)
    exponents[primitive] = scale * base_exponents[primitive];
  for (const auto atoms : {std::array<std::int32_t,4>{0,1,2,3}, {0,0,1,2},
                          {0,1,0,1}, {0,0,0,0}}) {
    double positions[12]{0.1,-0.2,-0.8,0.3,0.1,0.7,-0.5,0.6,0.2,0.8,-0.4,0.3};
    for (unsigned orientation = 0; orientation < 3U; ++orientation) {
      std::uint8_t ao_angular[12]{};
      for (unsigned center = 0; center < 4U; ++center)
        for (unsigned quantum = 0; quantum < shell_angular[center]; ++quantum)
          ++ao_angular[center * 3 + (center + quantum + orientation * (quantum + 1)) % 3];
      DeviceBatch batch{};
      batch.direct_nbf = 4;
      batch.positions = positions;
      batch.shell_atoms = atoms.data();
      batch.shell_angular = shell_angular;
      batch.shell_primitive_offsets = primitive_offsets;
      batch.primitive_exponents = exponents;
      batch.primitive_coefficients = coefficients;
      batch.direct_ao_shells = ao_shells;
      batch.direct_ao_angular = ao_angular;
      batch.direct_ao_coefficients = ao_coefficients;
      std::int32_t unique[4]{};
      unsigned centers = 0;
      for (auto atom : atoms) {
        bool found = false;
        for (unsigned center = 0; center < centers; ++center) found |= unique[center] == atom;
        if (!found) unique[centers++] = atom;
      }
      for (auto range : {CoulombRange::Full, CoulombRange::Long}) {
        for (double omega : {0.0, 0.37, 1e4}) {
          MaterializedDirectPairWork work{};
          const auto actual = contracted_direct_shared_response_gradient<order>(
              batch, 0, 0, 1, 2, 3, unique, centers, range, omega, &work);
          const unsigned live = centers > 1U ? 1U : 0U;
          if (work.bra_preparations != 4U * live || work.ket_preparations != 16U * live ||
              work.coulomb_preparations != 16U * live ||
              work.component_contractions != 16U * (centers - 1U) * live ||
              work.published_components != live) std::abort();
          for (unsigned center = 0; center + 1U < centers; ++center) {
            const auto retained = contracted_eri_cartesian_source_shell_class<
                First, Second, Third, Fourth, Dual3>(batch, 0, 1, 2, 3, 0, 1, 2, 3,
                    std::int64_t(unique[center]) * 3, range, omega);
            const double reference[3]{retained.derivative_x, retained.derivative_y,
                                      retained.derivative_z};
            for (unsigned axis = 0; axis < 3U; ++axis) {
              const double ad_error = std::abs(actual.center[center][axis] - reference[axis]);
              maximum_ad_error = std::max(maximum_ad_error, ad_error / (1.0 + std::abs(reference[axis])));
              if (!std::isfinite(actual.center[center][axis]) ||
                  ad_error > 3e-11 * (1.0 + std::abs(reference[axis]))) std::abort();
              const unsigned coordinate = 3U * unique[center] + axis;
              const double step = 1e-3 / std::sqrt(scale);
              const double original = positions[coordinate];
              positions[coordinate] = original + step;
              const auto plus = contracted_eri_cartesian_source_shell_class<
                  First, Second, Third, Fourth, double>(batch, 0, 1, 2, 3, 0, 1, 2, 3,
                      -1, range, omega);
              positions[coordinate] = original - step;
              const auto minus = contracted_eri_cartesian_source_shell_class<
                  First, Second, Third, Fourth, double>(batch, 0, 1, 2, 3, 0, 1, 2, 3,
                      -1, range, omega);
              positions[coordinate] = original + 2.0 * step;
              const auto plus_two = contracted_eri_cartesian_source_shell_class<
                  First, Second, Third, Fourth, double>(batch, 0, 1, 2, 3, 0, 1, 2, 3,
                      -1, range, omega);
              positions[coordinate] = original - 2.0 * step;
              const auto minus_two = contracted_eri_cartesian_source_shell_class<
                  First, Second, Third, Fourth, double>(batch, 0, 1, 2, 3, 0, 1, 2, 3,
                      -1, range, omega);
              positions[coordinate] = original;
              const double displaced = (8.0 * (plus - minus) - (plus_two - minus_two)) /
                                       (12.0 * step);
              const double error = std::abs(actual.center[center][axis] - displaced);
              maximum_displaced_error = std::max(maximum_displaced_error, error / (1.0 + std::abs(displaced)));
              if (error > 3e-7 * (1.0 + std::abs(displaced))) {
                std::fprintf(stderr, "order=%u range=%u omega=%g response=%g displaced=%g\n",
                    order, unsigned(range), omega, actual.center[center][axis], displaced);
                std::abort();
              }
              ++checks;
            }
          }
        }
      }
    }
  }
  }
}
int main() {
  qualify<3,2,2,0>();
  qualify<3,2,2,1>();
  qualify<3,2,2,2>();
  qualify<3,3,2,2>();
  qualify<3,3,3,2>();
  std::printf("shared atomic response and exact work PASS checks=%u normalized_AD=%g normalized_displaced=%g\n",
              checks, maximum_ad_error, maximum_displaced_error);
}
"""
