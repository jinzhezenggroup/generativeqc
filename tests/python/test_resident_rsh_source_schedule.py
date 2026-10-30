"""Execute the resident RSH adapter against destructive provider storage.

Manufactured derivatives check source coefficients, callback lifetimes and
failure propagation. Molecular force qualification supplies the independent
integral oracle; these fixtures do not pretend to evaluate integrals.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from _cpp_source_support import cpp_function_definition

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def resident_rsh_probe(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> Path:
    """Compile the production adapter with providers that invalidate each lease."""
    source = (ROOT / "src/scf/cuda_fock_execution.cpp").read_text()
    adapter = cpp_function_definition(
        source, "consume_prepared_cuda_direct_rsh_energy_derivatives_device"
    )
    directory = tmp_path_factory.mktemp("resident-rsh-source-schedule")
    cpp, binary = directory / "probe.cpp", directory / "probe"
    cpp.write_text(
        r"""
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include "scf/cuda_derivative_sources.hpp"
using namespace generativeqc::scf;
enum generativeqc_status { GENERATIVEQC_STATUS_SUCCESS,
  GENERATIVEQC_STATUS_NOT_IMPLEMENTED, GENERATIVEQC_STATUS_CUDA_ERROR };
enum class FockBackend { Cuda, Cpu };
enum class FockSpin { Restricted, Unrestricted };
enum class FockApproximation { Exact, DensityFitted };
enum class FockOperator { FullRange, LongRange };
struct Term {
  bool present{true};
  FockApproximation approximation{FockApproximation::Exact};
  FockOperator op{FockOperator::FullRange};
  double coefficient{1}, omega{};
};
struct Spec { unsigned derivative_order{}; FockSpin spin{}; Term coulomb, exchange; };
struct ResolvedFockBuild {
  FockBackend backend{FockBackend::Cuda}; Spec spec; double screening_tolerance{1e-12};
};
struct PreparedFockPlan {
  ResolvedFockBuild resolved;
  const ResolvedFockBuild& strategy() const { return resolved; }
  const PreparedFockPlan* cuda_direct_source() const { return this; }
};
struct Binding { size_t nbf{1}; explicit operator bool() const { return true; } };
Binding prepared_cuda_direct_derivative_binding(const PreparedFockPlan&) { return {}; }
std::array<double, 3> scratch{};
int full_status{}, range_status{};
bool full_submitted{}, range_submitted{};
void publish(CudaDerivativeSourceKind kind, size_t channels,
             const CudaDerivativeSourceConsumer& consumer) {
  consumer({kind, CudaDerivativeConvention::Force, 0, reinterpret_cast<void*>(1),
            scratch.data(), 1, channels});
  // No consumer may retain a pointer after its callback-scoped lease.
  scratch.fill(9999);
}
generativeqc_status consume_prepared_cuda_direct_shell_full_range_derivatives_device(
    const PreparedFockPlan& plan, const double*, const double*, size_t,
    const CudaDerivativeSourceConsumer& consumer, std::string&) {
  full_submitted = true;
  if (full_status) return GENERATIVEQC_STATUS_CUDA_ERROR;
  scratch = {2 * plan.resolved.spec.coulomb.coefficient,
             5 * plan.resolved.spec.exchange.coefficient, 0};
  publish(CudaDerivativeSourceKind::FullRangeJk, 2, consumer);
  return GENERATIVEQC_STATUS_SUCCESS;
}
generativeqc_status consume_cuda_direct_shell_rsh_energy_derivatives_device(
    const PreparedFockPlan*, FockSpin, double cj, double sr, double lr, double,
    const double*, const double*, size_t, const CudaDerivativeSourceConsumer& consumer,
    std::string&) {
  range_submitted = true;
  if (range_status) return GENERATIVEQC_STATUS_CUDA_ERROR;
  scratch = {2 * cj, (5 - 3) * sr, 3 * lr};
  publish(CudaDerivativeSourceKind::RangeSeparatedJk, 3, consumer);
  return GENERATIVEQC_STATUS_SUCCESS;
}
"""
        + adapter
        + r"""
int main(int argc, char** argv) {
  assert(argc == 4);
  const bool unrestricted = std::atoi(argv[1]);
  const int scenario = std::atoi(argv[2]);
  const double short_fraction = std::atof(argv[3]);
  PreparedFockPlan plan;
  auto& primary = plan.resolved;
  primary.spec.spin = unrestricted ? FockSpin::Unrestricted : FockSpin::Restricted;
  primary.spec.exchange.coefficient = -(unrestricted ? 1.0 : 0.5) * short_fraction;
  auto correction = primary;
  correction.spec.coulomb.present = false;
  correction.spec.exchange.op = FockOperator::LongRange;
  correction.spec.exchange.omega = scenario == 1 ? 0.4 : 0.3;
  correction.spec.exchange.coefficient = scenario == 2 ? 0.0
      : -(unrestricted ? 1.0 : 0.5) * (1.0 - short_fraction);
  if (scenario == 6)
    correction.spec.exchange.coefficient = std::numeric_limits<double>::denorm_min();
  if (scenario == 3) correction.screening_tolerance *= 2;
  if (scenario == 4) full_status = 1;
  if (scenario == 5) range_status = 1;
  const double sr = primary.spec.exchange.coefficient;
  const double delta = correction.spec.exchange.coefficient;
  std::array<double, 3> result{};
  bool full_received = false;
  unsigned callbacks = 0;
  const auto consume = [&](const CudaDerivativeSourceView& value) {
    ++callbacks;
    assert(value.device == 0 && value.producer_stream == reinterpret_cast<void*>(1));
    assert(value.convention == CudaDerivativeConvention::Force && value.coordinates == 1);
    if (value.kind == CudaDerivativeSourceKind::FullRangeJk) {
      assert(value.channels == 2 && !full_received);
      result[0] = value.values[0]; result[1] = value.values[1]; full_received = true;
    } else if (value.kind == CudaDerivativeSourceKind::RangeCorrection) {
      assert(value.channels == 1 && full_received);
      result[1] -= sr / delta * value.values[0];
      result[2] = (1 + sr / delta) * value.values[0];
    } else {
      assert(value.kind == CudaDerivativeSourceKind::RangeSeparatedJk && value.channels == 3);
      for (size_t i = 0; i < 3; ++i) result[i] = value.values[i];
    }
  };
  double density = 1;
  std::string detail;
  auto status = consume_prepared_cuda_direct_rsh_energy_derivatives_device(
      plan, correction, &density, unrestricted ? &density : nullptr, 1, consume, detail);
  if (scenario == 3) {
    assert(status == GENERATIVEQC_STATUS_NOT_IMPLEMENTED && callbacks == 0);
    assert(!full_submitted && !range_submitted);
  } else if (scenario == 4) {
    assert(status == GENERATIVEQC_STATUS_CUDA_ERROR && callbacks == 0);
    assert(full_submitted && !range_submitted);
  } else if (scenario == 5) {
    assert(status == GENERATIVEQC_STATUS_CUDA_ERROR && callbacks == 1);
  } else {
    assert(status == GENERATIVEQC_STATUS_SUCCESS);
    const std::array<double, 3> expected{2, 2 * sr, 3 * (sr + delta)};
    for (size_t i = 0; i < 3; ++i) assert(std::abs(result[i] - expected[i]) < 1e-13);
    // Generic omega and vanishing correction retain the fused bounded source.
    assert(full_received == !(scenario == 1 || scenario == 2 || !std::isfinite(sr / delta)));
  }
}
"""
    )
    native_cxx.build_executable(
        [cpp], binary, compile_args=["-std=c++17", "-O2", f"-I{ROOT / 'src'}"]
    )
    return binary


@pytest.mark.parametrize("unrestricted", (False, True))
@pytest.mark.parametrize("scenario", range(7))
@pytest.mark.parametrize("short_fraction", (0.0, 0.15, 0.8))
def test_resident_rsh_sources_and_failure_boundaries(
    resident_rsh_probe: Path, unrestricted: bool, scenario: int, short_fraction: float
) -> None:
    """Keep both spins, zero SR, generic omega and failed providers explicit."""
    subprocess.run(
        [
            str(resident_rsh_probe),
            str(int(unrestricted)),
            str(scenario),
            str(short_fraction),
        ],
        check=True,
        timeout=10,
    )
