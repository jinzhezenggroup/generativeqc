/** Validation-only entry into the complete production bounded RHF E+F solve.
 * No reference integrals, alternate SCF loop or derivative evaluator lives here.
 * Source construction supplies normalized metadata; timing includes production
 * executable preparation, SCF, physical-reference validation and native forces.
 */
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>

#include "hf/rhf_frame_response.hpp"
#include "molecule/nuclear_gradient.hpp"
#include "posthf/raw_source.hpp"
#include "runtime/cuda_resources.cuh"
#include "scf/mean_field.hpp"

namespace {
using namespace generativeqc;
struct Endpoint {
  core::System system;
  scf::ScfOptions options;
  scf::CudaRhfBucketPlan* plan{};
  std::vector<double> cold_density, moved_density;
  bool moved{};
  ~Endpoint() { scf::destroy_rhf_cuda_bucket_plan(plan); }
};
void message(char* error, std::size_t capacity, const std::exception& failure) {
  if (error && capacity) std::snprintf(error, capacity, "%s", failure.what());
}
}  // namespace

extern "C" const char* selective_eri_rhf_native_library() noexcept {
  Dl_info identity{};
  return dladdr(reinterpret_cast<void*>(&scf::run_rhf_cuda_cached), &identity) ? identity.dli_fname
                                                                               : nullptr;
}

/** Freeze the complete numeric admission and strict physical equation. */
extern "C" int selective_eri_rhf_create(void* raw, std::size_t budget, unsigned iterations,
                                        double tolerance, void** output, char* error,
                                        std::size_t error_capacity) noexcept {
  if (output) *output = nullptr;
  try {
    if (!raw || !output || !budget || !iterations || !std::isfinite(tolerance) || tolerance <= 0 ||
        !std::getenv("SLURM_JOB_ID"))
      throw std::invalid_argument("RHF endpoint requires bounded Slurm admission");
    auto endpoint = std::make_unique<Endpoint>();
    endpoint->system = static_cast<posthf::RawSource*>(raw)->orbital();
    endpoint->options.max_iterations = iterations;
    endpoint->options.energy_tolerance = tolerance;
    endpoint->options.density_tolerance = tolerance;
    endpoint->options.screening_tolerance = 0;
    endpoint->options.precision_mode = GENERATIVEQC_PRECISION_FP64;
    endpoint->options.export_physical_reference = true;
    endpoint->options.reference_memory_budget_bytes = budget;
    endpoint->options.compute_forces = false;
    *output = endpoint.release();
    return 0;
  } catch (const std::exception& failure) {
    message(error, error_capacity, failure);
    return -1;
  }
}

/** Phases: fresh cold, fixed post-cold warm, moved, fixed post-moved warm.
 * Move uses the same executable topology and an explicit 0.05-Bohr displacement.
 * Failed/nonconverged results never publish energies or force destinations.
 */
extern "C" int selective_eri_rhf_run(void* opaque, unsigned phase, double* energy, double* forces,
                                     std::size_t force_elements, std::uint64_t* counts,
                                     double* residuals, char* error,
                                     std::size_t error_capacity) noexcept {
  try {
    if (!opaque || phase > 3 || !energy || !forces || !counts || !residuals)
      throw std::invalid_argument("invalid RHF endpoint destination");
    auto& endpoint = *static_cast<Endpoint*>(opaque);
    if (force_elements != 3 * endpoint.system.atoms.size() ||
        (phase && endpoint.cold_density.empty()) || (phase == 3 && !endpoint.moved) ||
        (phase == 2 && endpoint.moved) || (phase < 2 && endpoint.moved))
      throw std::invalid_argument("invalid RHF endpoint phase or shape");
    runtime::CudaDeviceScope device(0);
    if (phase == 2) {
      endpoint.system.atoms.front().position[0] += 0.05;
      endpoint.moved = true;
    }
    const auto* seed = phase == 0   ? nullptr
                       : phase == 3 ? &endpoint.moved_density
                                    : &endpoint.cold_density;
    bool reused = false;
    const auto result = scf::run_rhf_cuda_cached(&endpoint.plan, endpoint.system, endpoint.options,
                                                 0, seed, &reused);
    if (!result.converged || !result.reference || !std::isfinite(result.energy))
      throw std::runtime_error("RHF endpoint did not publish a finite physical reference");
    const auto dimension = result.reference->nbf;
    std::vector<double> zero_seed(dimension * dimension);
    hf::RHFFrameResponseOptions response_options;
    response_options.maximum_bytes = endpoint.options.reference_memory_budget_bytes;
    // Charge the exported SCF peak conservatively while its cached plan remains
    // alive. The response owner additionally charges its reference and seeds.
    response_options.caller_bytes = result.reference->numeric_capacity_bytes;
    response_options.profile_jk = true;
    auto response = hf::rhf_frame_response_cuda(endpoint.system, *result.reference, zero_seed,
                                                zero_seed, 0, response_options);
    auto gradient = std::move(response.gradient);
    molecule::add_nuclear_repulsion_gradient(endpoint.system, gradient);
    if (gradient.size() != force_elements ||
        !std::all_of(gradient.begin(), gradient.end(),
                     [](double value) { return std::isfinite(value); }))
      throw std::runtime_error("RHF endpoint did not publish finite complete forces");
    if (phase == 0) endpoint.cold_density = result.reference->density;
    if (phase == 2) endpoint.moved_density = result.reference->density;
    *energy = result.energy;
    std::transform(gradient.begin(), gradient.end(), forces, [](double value) { return -value; });
    counts[0] = result.iterations;
    counts[1] = result.precision.strict_stage_fock_builds +
                result.precision.mixed_stage_fock_builds + result.precision.post_scf_fock_builds;
    counts[2] = reused;
    counts[3] = response.jk_actions;
    counts[4] = response.derivative_passes;
    counts[5] = response.orbital_actions;
    counts[6] = response.jk_quartet_visits;
    counts[7] = response.jk_eri_evaluations;
    counts[8] = response.resident_jk_bytes;
    residuals[0] = result.energy_change;
    residuals[1] = result.density_rms;
    residuals[2] = result.physical_residual_rms;
    return 0;
  } catch (const std::exception& failure) {
    message(error, error_capacity, failure);
    return -1;
  }
}

extern "C" void selective_eri_rhf_destroy(void* opaque) noexcept {
  delete static_cast<Endpoint*>(opaque);
}
