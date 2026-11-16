#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "molecule/basis.hpp"
#include "runtime/resource_ledger.hpp"
#include "scf/cuda/rhf_bucket_internal.hpp"
#include "scf/cuda_batch.hpp"

namespace {
using namespace generativeqc;

/** Match the independent s-only and spherical s/p/d profile qualification
 * fixtures. This runner measures complete endpoints, not isolated kernels.
 */
core::System make_system(bool unrestricted, bool polarized) {
  core::System system;
  system.atoms = {{2, {0., 0., -0.60}}, {1, {0., 0., 0.85}}};
  system.shells = {
      {0, 0, {{6.36242139, 0.15432897}, {1.15892300, 0.53532814}, {0.31364979, 0.44463454}}},
      {1, 0, {{3.42525091, 0.15432897}, {0.62391373, 0.53532814}, {0.16885540, 0.44463454}}}};
  if (polarized) {
    system.shells.push_back({0, 1, {{0.8, 1.0}}});
    system.shells.push_back({0, 2, {{0.6, 1.0}}});
    system.shells.push_back({1, 1, {{0.4, 1.0}}});
    system.shells.push_back({1, 2, {{0.3, 1.0}}});
  }
  system.charge = unrestricted ? 0 : 1;
  system.multiplicity = unrestricted ? 2 : 1;
  system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
  std::string detail;
  if (molecule::validate_and_normalize(system, detail) != GENERATIVEQC_STATUS_SUCCESS)
    throw std::runtime_error(detail);
  return system;
}

struct Plan {
  scf::CudaRhfBucketPlan* value{};
  ~Plan() { scf::destroy_rhf_cuda_bucket_plan(value); }
};

/** Each repetition owns fresh preparation; warm and moved calls retain only
 * that owner's plan and the immediately preceding physical density. Owned
 * bytes are an allocation receipt, explicitly not a sampled memory peak.
 */
void measure(bool unrestricted, bool polarized, bool forces, std::size_t batch_size,
             unsigned repetition) {
  auto system = make_system(unrestricted, polarized);
  scf::ScfOptions options;
  options.max_iterations = 100;
  options.energy_tolerance = 1e-12;
  options.density_tolerance = 1e-10;
  options.screening_tolerance = 1e-12;
  options.precision_mode = GENERATIVEQC_PRECISION_FP64;
  options.compute_forces = forces;
  auto ledger = std::make_shared<runtime::DeviceResourceLedger>();
  ledger->device = 0;
  ledger->limit = std::numeric_limits<std::size_t>::max();
  runtime::DeviceResourceScope scope(ledger);
  Plan owner;
  std::vector<double> density;
  for (const char* phase : {"cold", "warm", "moved-warm"}) {
    if (phase[0] == 'm') system.atoms[1].position[2] += 0.03;
    {
      std::lock_guard<std::mutex> lock(runtime::device_resource_mutex);
      ledger->peak = ledger->live;
    }
    const auto start = std::chrono::steady_clock::now();
    const std::vector<core::System> systems(batch_size, system);
    const std::vector<const std::vector<double>*> densities(batch_size,
                                                            density.empty() ? nullptr : &density);
    const auto rows =
        unrestricted
            ? scf::run_uhf_cuda_bucket_cached(&owner.value, systems, options, densities, 0)
            : scf::run_rhf_cuda_bucket_cached(&owner.value, systems, options, densities, 0);
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (rows.size() != batch_size || !owner.value)
      throw std::runtime_error("endpoint lost its prepared batch");
    std::size_t iterations = 0, fock_builds = 0;
    for (const auto& row : rows) {
      if (row.status != GENERATIVEQC_STATUS_SUCCESS || !row.scf.converged ||
          !std::isfinite(row.scf.energy) || !row.scf.executed_fock_builds)
        throw std::runtime_error("endpoint did not complete its physical SCF work");
      iterations += row.scf.iterations;
      fock_builds += *row.scf.executed_fock_builds;
    }
    std::cout << std::setprecision(17) << "{\"spin\":\"" << (unrestricted ? "UHF" : "RHF")
              << "\",\"nbf\":" << molecule::ao_count(system) << ",\"forces\":" << forces
              << ",\"batch\":" << batch_size << ",\"repetition\":" << repetition << ",\"phase\":\""
              << phase << "\",\"endpoint_ms\":" << elapsed_ms << ",\"iterations\":" << iterations
              << ",\"fock_builds\":" << fock_builds
              << ",\"owned_device_bytes\":" << scf::hf_cuda_owned_device_bytes(owner.value)
              << ",\"tracked_numeric_peak_bytes\":" << ledger->peak << ",\"matrix_provider\":\""
              << (owner.value->resources.matrix_binding_->requires_provider_context() ? "cublas"
                                                                                      : "native")
              << "\",\"persistent_eri\":" << owner.value->persistent_eri
              << ",\"energy\":" << rows.front().scf.energy << ",\"force_values\":[";
    for (std::size_t element = 0; element < rows.front().scf.forces.size(); ++element) {
      if (element) std::cout << ',';
      std::cout << rows.front().scf.forces[element];
    }
    std::cout << "]}\n";
    density = rows.front().scf.density;
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    unsigned repetitions = 3;
    if (argc > 2) throw std::runtime_error("usage: direct_jk_policy_benchmark [repetitions]");
    if (argc == 2) {
      char* end = nullptr;
      const auto parsed = std::strtoul(argv[1], &end, 10);
      if (!end || *end || parsed == 0 || parsed > 100)
        throw std::runtime_error("repetitions must be in [1, 100]");
      repetitions = static_cast<unsigned>(parsed);
    }
    if (std::getenv("GENERATIVEQC_HF_MATRIX_PROFILE"))
      throw std::runtime_error("default qualification requires no tuning profile");
    for (unsigned repetition = 0; repetition < repetitions; ++repetition)
      for (bool polarized : {false, true})
        for (bool unrestricted : {false, true})
          for (bool forces : {false, true})
            for (std::size_t batch_size : {1U, 2U})
              measure(unrestricted, polarized, forces, batch_size, repetition);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
