"""Exercise the native diagnostic publication with distinguishable host values.

This shim tests publication, not CUDA execution or force equations. The existing
native force gates and opt-in public finite-difference test cover those owners.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_completed_df_force_publishes_actual_response(
    tmp_path: Path, native_cxx: object
) -> None:
    source = (ROOT / "src/methods/df_ccsdt_force.cu").read_text()
    helper_start = source.index("void publish_force_diagnostic(")
    # Compile only this helper, not unrelated helpers added to its namespace.
    helper_end = source.index("\n}", helper_start) + 2
    helper = source[helper_start:helper_end]
    tail_start = source.index("  if (demand.mp2 || demand.companions) {")
    tail_end = source.index("\n}\n", tail_start)
    tail = source[tail_start:tail_end]
    header = (ROOT / "src/methods/df_ccsdt_force.hpp").read_text()
    demand_start = header.index("struct DFCorrelationDemand {")
    demand_end = header.index("\n};", demand_start) + 3
    demand = header[demand_start:demand_end]
    # Energy-only requests must return before force diagnostics are published.
    assert source.index("if (!forces)") < tail_start
    owner = (ROOT / "src/methods/df_rccsdt_method.cpp").read_text()
    assert "last_correlation_ = native.correlation;" in owner

    unit = tmp_path / "df-force-publication.cpp"
    unit.write_text(
        r"""
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "generativeqc/generativeqc.h"
#include "posthf/capacity.hpp"
using generativeqc::posthf::checked_add;
std::size_t bytes(std::size_t n) {
  return generativeqc::posthf::checked_mul(n, sizeof(double));
}
struct Response {
  std::size_t iterations{11}, restarts{3}, workspace_bytes{4096};
  std::size_t measured_workspace_peak_bytes{3072}, workspace_allocation_count{9};
  double residual_norm{2e-12}, relative_residual{3e-13};
  bool accepted{true};
  bool converged() const { return accepted; }
};
struct DFCCSDTResult {
  generativeqc_correlation_diagnostic correlation{};
  struct {
    Response orbital_response;
    double orbital_residual{7e-12};
    std::string operator_hash{"qualified-orbital-operator"};
    std::size_t numeric_capacity_bytes{8192};
  } orbital;
  std::size_t numeric_capacity_bytes{8192};
  std::vector<double> forces{1, 2, 3};
  struct { std::vector<double> forces; } method_result;
  double total_seconds{};
};
double elapsed(double) { return 1.0; }
"""
        + demand
        + helper
        + r"""
// Keep the detached result visible so refusal cannot silently publish a force
// vector or diagnostics before complete numeric admission succeeds.
DFCCSDTResult finish(DFCCSDTResult& result, DFCorrelationDemand demand = {},
                    std::size_t maximum_bytes = 16384) {
  double started = 0;
  const auto coords = result.forces.size();
  const struct { std::size_t maximum_bytes; } orbital_options{maximum_bytes};
"""
        + tail
        + r"""
}
int main() {
  DFCCSDTResult native;
  native.correlation.reference_energy = -1.25;
  native.correlation.ccsd_iterations = 17;
  native.correlation.ccsd_t_triples_energy = -0.125;
  std::fill_n(native.correlation.response_operator_hash, 65, 'x');
  auto result = finish(native);
  const auto& diagnostic = result.correlation;
  if (result.method_result.forces != native.forces ||
      diagnostic.response_iterations != 11 || diagnostic.response_restarts != 3 ||
      diagnostic.response_absolute_residual != 7e-12 ||
      diagnostic.response_relative_residual != 3e-13 ||
      diagnostic.response_workspace_bytes != 4096 ||
      diagnostic.measured_response_workspace_peak_bytes != 3072 ||
      diagnostic.response_workspace_allocation_count != 9 ||
      diagnostic.numeric_capacity_bytes != 8192 ||
      diagnostic.planned_endpoint_peak_bytes != 8192 ||
      diagnostic.force_provenance_flags != 0xf ||
      std::strcmp(diagnostic.response_operator_hash, "qualified-orbital-operator") != 0)
    return 1;
  // Admission bounds must not be relabeled as measurements, nor overwrite
  // the separate primal/triples diagnostics.
  if (diagnostic.measured_endpoint_peak_bytes || diagnostic.derivative_workspace_bytes ||
      diagnostic.reference_energy != -1.25 || diagnostic.ccsd_iterations != 17 ||
      diagnostic.ccsd_t_triples_energy != -0.125)
    return 2;
  native.orbital.orbital_residual = 1e-12;
  native.orbital.operator_hash = std::string(80, 'h');
  result = finish(native);
  if (result.correlation.response_absolute_residual != 2e-12 ||
      std::strlen(result.correlation.response_operator_hash) != 64)
    return 3;
  // A converged initial-residual H2-like solve legitimately takes zero steps.
  native.orbital.orbital_response.iterations = 0;
  native.orbital.orbital_response.restarts = 0;
  native.orbital.orbital_response.residual_norm = 0;
  native.orbital.orbital_residual = 0;
  result = finish(native);
  if (result.correlation.response_iterations || result.correlation.response_restarts ||
      result.correlation.response_absolute_residual ||
      result.correlation.force_provenance_flags != 0xf)
    return 4;
  // The internal fixed-frame mode must not claim a converged orbital solve.
  native.orbital.orbital_response.accepted = false;
  if (finish(native).correlation.force_provenance_flags != 0xe) return 5;

  // Both standalone MP2 and grouped companions charge the coexisting public
  // force vector. Only an MP2 primary omits the triples provenance bit.
  const DFCorrelationDemand demands[] = {
      {true, 0}, {true, 1}, {false, 1}, {false, 2}, {false, 3}};
  for (const auto demand : demands) {
    for (const bool converged : {false, true}) {
      native = DFCCSDTResult{};
      native.orbital.orbital_response.accepted = converged;
      native.method_result.forces = {-10, -20, -30};
      native.correlation.response_iterations = 97;
      native.correlation.numeric_capacity_bytes = 1234;
      native.correlation.planned_endpoint_peak_bytes = 4567;
      native.correlation.force_provenance_flags = 0xa0;
      std::strcpy(native.correlation.response_operator_hash, "unpublished");
      const auto unpublished_forces = native.method_result.forces;
      const auto publication_peak = 8192 + 3 * sizeof(double);
      bool refused = false;
      try {
        (void)finish(native, demand, publication_peak - 1);
      } catch (const std::length_error&) {
        refused = true;
      }
      if (!refused || native.method_result.forces != unpublished_forces ||
          native.forces != std::vector<double>({1, 2, 3}) ||
          native.numeric_capacity_bytes != 8192 ||
          native.correlation.response_iterations != 97 ||
          native.correlation.numeric_capacity_bytes != 1234 ||
          native.correlation.planned_endpoint_peak_bytes != 4567 ||
          native.correlation.force_provenance_flags != 0xa0 ||
          std::strcmp(native.correlation.response_operator_hash, "unpublished") != 0)
        return 6;
      result = finish(native, demand, publication_peak);
      const auto expected_flags = (converged ? 0xfU : 0xeU) & (demand.mp2 ? 0x7U : 0xfU);
      if (result.method_result.forces != native.forces ||
          result.correlation.response_iterations != 11 ||
          result.correlation.response_absolute_residual != 7e-12 ||
          result.correlation.force_provenance_flags != expected_flags ||
          result.numeric_capacity_bytes != publication_peak ||
          result.correlation.numeric_capacity_bytes != publication_peak ||
          result.correlation.planned_endpoint_peak_bytes != publication_peak ||
          result.correlation.measured_endpoint_peak_bytes ||
          result.correlation.derivative_workspace_bytes)
        return 7;
      // An earlier phase's larger admitted peak must survive publication.
      native.numeric_capacity_bytes = 10000;
      result = finish(native, demand, 10000);
      if (result.correlation.numeric_capacity_bytes != 10000 ||
          result.correlation.planned_endpoint_peak_bytes != 10000)
        return 8;
    }
  }
}
"""
    )
    executable = native_cxx.build_executable(
        [unit],
        tmp_path / "publication",
        compile_args=(
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{ROOT / 'include'}",
            f"-I{ROOT / 'src'}",
        ),
    )
    subprocess.run([str(executable)], check=True, timeout=10)
