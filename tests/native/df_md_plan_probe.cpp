// Exercise the actual source/metric/J handoff against independent Python oracles.
// This probe deliberately bypasses high-level admission so small bases can cover
// streamed execution, explicit budget refusal, and changed-input graph replay.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "molecule/basis.hpp"
#include "scf/cuda/df_jk_internal.hpp"
#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda_density_fitting_integrals.hpp"
#include "scf/cuda_density_fitting_md.hpp"

using namespace generativeqc;
using namespace generativeqc::scf;
namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool value, const std::string& detail) {
  if (!value) throw std::runtime_error(detail);
}
core::System read_system() {
  core::System result;
  std::size_t atoms{}, shells{};
  int spherical{};
  std::cin >> atoms >> shells >> result.charge >> spherical;
  if (!atoms || atoms > 8 || !shells || shells > 16)
    throw std::runtime_error("fixture exceeds small-basis qualification bound");
  result.basis_representation =
      spherical ? GENERATIVEQC_BASIS_SPHERICAL : GENERATIVEQC_BASIS_CARTESIAN;
  for (std::size_t i = 0; i < atoms; ++i) {
    core::Atom atom;
    std::cin >> atom.atomic_number >> atom.position[0] >> atom.position[1] >> atom.position[2];
    result.atoms.push_back(atom);
  }
  for (std::size_t i = 0; i < shells; ++i) {
    core::Shell shell;
    std::size_t primitives{};
    std::cin >> shell.atom_index >> shell.angular_momentum >> primitives;
    if (!primitives || primitives > 16 || shell.angular_momentum > 4)
      throw std::runtime_error("fixture exceeds primitive/angular bound");
    shell.primitives.resize(primitives);
    for (auto& primitive : shell.primitives)
      std::cin >> primitive.exponent >> primitive.coefficient;
    result.shells.push_back(shell);
  }
  std::string detail;
  if (!std::cin || molecule::validate_and_normalize(result, detail) != GENERATIVEQC_STATUS_SUCCESS)
    throw std::runtime_error(detail.empty() ? "invalid basis fixture" : detail);
  return result;
}

}  // namespace
int main(int argc, char** argv) {
  try {
    require(argc == 2 && std::getenv("SLURM_JOB_ID"), "use a finite Slurm allocation");
    const auto orbital = read_system(), auxiliary = read_system();
    CudaDensityFittingIntegralSource* source{};
    std::vector<double> metric;
    std::size_t n{}, a{};
    std::string detail;
    require(
        create_cuda_density_fitting_integral_source(0, {orbital}, {auxiliary}, &source, metric, n,
                                                    a, detail) == GENERATIVEQC_STATUS_SUCCESS,
        detail);
    std::vector<double> density(n * n), result(5 * n * n), exchange;
    for (auto& value : density) std::cin >> value;
    require(bool(std::cin), "missing input density");
    CudaDensityFittingJkPlan* raw{};
    std::vector<CudaDensityFittingMetricDiagnostic> diagnostics;
    // A partial AO row extent is essential: a full A/Q tile is resident even
    // when retain_three_center is false, and correctly refuses this candidate.
    require(create_cuda_density_fitting_jk_plan_from_source(0, &source, 1, n, a, metric, 1e-12, a,
                                                            n, &raw, diagnostics,
                                                            detail) == GENERATIVEQC_STATUS_SUCCESS,
            detail);
    std::unique_ptr<CudaDensityFittingJkPlan, decltype(&destroy_cuda_density_fitting_jk_plan)> plan(
        raw, &destroy_cuda_density_fitting_jk_plan);
    std::vector<double> incumbent;
    require(execute_cuda_density_fitting_rhf_jk(plan.get(), density, incumbent, exchange, detail,
                                                {true, false}) == GENERATIVEQC_STATUS_SUCCESS,
            detail);
    std::copy(incumbent.begin(), incumbent.end(), result.begin());
    const auto original = diagnostics.front();
    // First refuse before owner allocation, then refuse inside layout preflight.
    // Both attempts must leave an executable incumbent and unchanged accounting.
    constexpr std::size_t cap = 2ULL << 30;
    for (const auto slack : {0U, 1U}) {
      require(try_install_cuda_df_md_coulomb(plan.get(), orbital, auxiliary,
                                             original.peak_device_bytes + slack, cap, diagnostics,
                                             detail) == GENERATIVEQC_STATUS_SUCCESS,
              detail);
      require(!diagnostics.front().md_coulomb && !plan->md_coulomb &&
                  diagnostics.front().device_resident_bytes == original.device_resident_bytes,
              "capacity refusal mutated owner/accounting");
      std::vector<double> after;
      require(execute_cuda_density_fitting_rhf_jk(plan.get(), density, after, exchange, detail,
                                                  {true, false}) == GENERATIVEQC_STATUS_SUCCESS,
              detail);
      for (std::size_t i = 0; i < after.size(); ++i)
        require(std::abs(after[i] - incumbent[i]) < 1e-10, "refusal changed incumbent J");
    }
    require(try_install_cuda_df_md_coulomb(plan.get(), orbital, auxiliary, cap, cap, diagnostics,
                                           detail) == GENERATIVEQC_STATUS_SUCCESS,
            detail);
    const bool admitted = diagnostics.front().effective_rank == a;
    require(diagnostics.front().md_coulomb == admitted && bool(plan->md_coulomb) == admitted,
            "full-rank admission / truncated-metric fallback disagreed: " + detail);
    require(diagnostics.front().device_resident_bytes ==
                    original.device_resident_bytes + diagnostics.front().md_coulomb_device_bytes &&
                diagnostics.front().peak_device_bytes <= cap,
            "owner is not charged to the enclosing budget");
    if (admitted)
      require(cuda_df_md_coulomb_reservation(orbital, auxiliary, cap) ==
                  diagnostics.front().md_coulomb_device_bytes,
              "topology reservation differs from the installed owner footprint");
    if (admitted)
      require(try_install_cuda_df_md_coulomb(plan.get(), orbital, auxiliary, cap, cap, diagnostics,
                                             detail) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
              "duplicate owner installation was accepted");
    // Reuse plan-owned density/output storage; capture contains the same public
    // charge/metric/projection route used by ordinary SCF. No test-only math.
    const auto bytes = n * n * sizeof(double);
    check(cudaMemcpyAsync(plan->primary_density, density.data(), bytes, cudaMemcpyHostToDevice,
                          plan->stream));
    require(
        cuda_df::build_coulomb(*plan, plan->primary_density, detail) == GENERATIVEQC_STATUS_SUCCESS,
        detail);
    check(cudaMemcpyAsync(result.data() + n * n, plan->coulomb, bytes, cudaMemcpyDeviceToHost,
                          plan->stream));
    check(cudaStreamSynchronize(plan->stream));
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    if (admitted) {
      check(cudaStreamBeginCapture(plan->stream, cudaStreamCaptureModeThreadLocal));
      require(cuda_df::build_coulomb(*plan, plan->primary_density, detail) ==
                  GENERATIVEQC_STATUS_SUCCESS,
              detail);
      check(cudaStreamEndCapture(plan->stream, &graph));
      check(cudaGraphInstantiate(&executable, graph, 0));
    }
    const auto initial = density;
    for (std::size_t replay = 0; replay < 3; ++replay) {
      const double scale = replay == 0 ? 1 : replay == 1 ? 0 : -0.37;
      for (std::size_t i = 0; i < density.size(); ++i) density[i] = scale * initial[i];
      check(cudaMemcpyAsync(plan->primary_density, density.data(), bytes, cudaMemcpyHostToDevice,
                            plan->stream));
      if (admitted)
        check(cudaGraphLaunch(executable, plan->stream));
      else
        require(cuda_df::build_coulomb(*plan, plan->primary_density, detail) ==
                    GENERATIVEQC_STATUS_SUCCESS,
                detail);
      check(cudaMemcpyAsync(result.data() + (replay + 2) * n * n, plan->coulomb, bytes,
                            cudaMemcpyDeviceToHost, plan->stream));
      check(cudaStreamSynchronize(plan->stream));
    }
    if (admitted) {
      check(cudaGraphExecDestroy(executable));
      check(cudaGraphDestroy(graph));
    }
    std::ofstream output(argv[1], std::ios::binary);
    output.write(reinterpret_cast<const char*>(result.data()), result.size() * sizeof(double));
    output.write(reinterpret_cast<const char*>(metric.data()), metric.size() * sizeof(double));
    require(bool(output), "failed to write metric/J evidence");
    std::cout << "{\"nbf\":" << n << ",\"naux\":" << a
              << ",\"admitted\":" << (admitted ? "true" : "false")
              << ",\"owner_device_bytes\":" << diagnostics.front().md_coulomb_device_bytes
              << ",\"peak_device_bytes\":" << diagnostics.front().peak_device_bytes
              << ",\"host_peak_bytes\":" << diagnostics.front().peak_host_bytes << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
