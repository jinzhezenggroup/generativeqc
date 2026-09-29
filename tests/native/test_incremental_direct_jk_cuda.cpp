#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "generativeqc/generativeqc.h"
#include "molecule/basis.hpp"
#include "scf/cuda_batch.hpp"
#include "scf/mean_field.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

bool cuda_device_available() {
  generativeqc_context_descriptor descriptor{sizeof(generativeqc_context_descriptor),
                                             GENERATIVEQC_ABI_VERSION, 0,
                                             GENERATIVEQC_BACKEND_CUDA};
  generativeqc_context* context = nullptr;
  const generativeqc_status status = generativeqc_context_create(&descriptor, &context);
  if (context != nullptr) generativeqc_context_destroy(context);
  return status == GENERATIVEQC_STATUS_SUCCESS;
}

generativeqc::core::System asymmetric_two_center(bool unrestricted) {
  generativeqc::core::System system;
  system.atoms = {
      {2, {0.0, 0.0, -0.60}},
      {1, {0.0, 0.0, 0.85}},
  };
  system.shells = {
      {0, 0, {{6.36242139, 0.15432897}, {1.15892300, 0.53532814}, {0.31364979, 0.44463454}}},
      {1, 0, {{3.42525091, 0.15432897}, {0.62391373, 0.53532814}, {0.16885540, 0.44463454}}},
  };
  system.charge = unrestricted ? 0 : 1;
  system.multiplicity = unrestricted ? 2 : 1;
  system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
  std::string detail;
  require(
      generativeqc::molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
      "incremental Direct-J/K fixture normalization failed");
  return system;
}

generativeqc::scf::RhfBucketItem run_cached(generativeqc::scf::CudaRhfBucketPlan** plan,
                                            const generativeqc::core::System& system,
                                            const generativeqc::scf::ScfOptions& options,
                                            bool unrestricted) {
  const std::vector<generativeqc::core::System> systems{system};
  const std::vector<const std::vector<double>*> cold{nullptr};
  const auto rows =
      unrestricted
          ? generativeqc::scf::run_uhf_cuda_bucket_cached(plan, systems, options, cold, 0, false)
          : generativeqc::scf::run_rhf_cuda_bucket_cached(plan, systems, options, cold, 0, false);
  require(rows.size() == 1, "incremental Direct-J/K CUDA bucket returned wrong size");
  return rows.front();
}

void compare_final_state(const generativeqc::scf::ScfResult& baseline,
                         const generativeqc::scf::ScfResult& incremental, double tolerance) {
  require(baseline.converged && incremental.converged,
          "incremental Direct-J/K CUDA parity fixture did not converge");
  require(std::abs(baseline.energy - incremental.energy) < tolerance,
          "incremental Direct-J/K changed final energy");
  require(baseline.forces.size() == incremental.forces.size(),
          "incremental Direct-J/K changed force shape");
  for (std::size_t i = 0; i < baseline.forces.size(); ++i) {
    require(std::abs(baseline.forces[i] - incremental.forces[i]) < 10.0 * tolerance,
            "incremental Direct-J/K changed final forces");
  }
}

void verify_case(bool unrestricted, double screening_tolerance, unsigned requested_interval) {
  const auto system = asymmetric_two_center(unrestricted);
  generativeqc::scf::ScfOptions baseline_options;
  baseline_options.max_iterations = 100;
  baseline_options.energy_tolerance = 1.0e-12;
  baseline_options.density_tolerance = 1.0e-10;
  baseline_options.screening_tolerance = screening_tolerance;
  baseline_options.precision_mode = GENERATIVEQC_PRECISION_FP64;
  baseline_options.compute_forces = true;

  generativeqc::scf::CudaRhfBucketPlan* plan = nullptr;
  const auto baseline = run_cached(&plan, system, baseline_options, unrestricted);
  require(baseline.status == GENERATIVEQC_STATUS_SUCCESS && baseline.scf.converged,
          "baseline CUDA Direct-J/K fixture failed");

  auto incremental_options = baseline_options;
  incremental_options.incremental_direct_jk = true;
  incremental_options.incremental_direct_jk_rebuild_interval = requested_interval;
  const auto incremental = run_cached(&plan, system, incremental_options, unrestricted);
  require(incremental.status == GENERATIVEQC_STATUS_SUCCESS && incremental.scf.converged,
          "incremental CUDA Direct-J/K fixture failed");
  compare_final_state(baseline.scf, incremental.scf, screening_tolerance == 0.0 ? 5.0e-10 : 5.0e-8);

  const auto& diagnostic = incremental.scf.incremental_direct_jk;
  require(diagnostic.requested && diagnostic.active,
          "native CUDA did not activate requested incremental Direct-J/K");
  const std::uint64_t builds = diagnostic.anchor_full_builds + diagnostic.delta_builds;
  require(builds == incremental.scf.iterations && diagnostic.delta_builds != 0U,
          "native CUDA incremental build counters do not match SCF iterations");
  require(diagnostic.anchor_updates == diagnostic.delta_builds,
          "native CUDA incremental anchor-update semantics changed");
  require(diagnostic.max_abs_delta_density > 0.0,
          "native CUDA incremental path did not observe a density delta");
  require(diagnostic.post_scf_full_builds != 0U,
          "native CUDA incremental path skipped strict final full rebuilds");
  require(diagnostic.quartet_work_counters_valid,
          "fixed-topology incremental Direct-J/K work counters are not valid");
  require(diagnostic.full_admitted_shell_quartets <= diagnostic.full_candidate_shell_quartets &&
              diagnostic.delta_admitted_shell_quartets <=
                  diagnostic.delta_candidate_shell_quartets,
          "incremental Direct-J/K admitted more shell quartets than it visited");
  require(diagnostic.full_admitted_quartet_tiles >= diagnostic.full_admitted_shell_quartets &&
              diagnostic.delta_admitted_quartet_tiles >= diagnostic.delta_admitted_shell_quartets,
          "incremental Direct-J/K tile census is smaller than its admitted shell-quartet census");
  require(diagnostic.full_candidate_shell_quartets ==
                  diagnostic.full_admitted_shell_quartets +
                      diagnostic.full_rejected_shell_quartets &&
              diagnostic.delta_candidate_shell_quartets ==
                  diagnostic.delta_admitted_shell_quartets +
                      diagnostic.delta_rejected_shell_quartets,
          "incremental Direct-J/K candidate/admitted/rejected counters do not reconcile");

  if (screening_tolerance == 0.0) {
    require(diagnostic.anchor_full_builds == 1U && diagnostic.periodic_rebuilds == 0U,
            "unscreened exact incremental mode rebuilt unexpectedly");
    require(diagnostic.full_rejected_shell_quartets == 0U &&
                diagnostic.delta_rejected_shell_quartets == 0U,
            "unscreened incremental Direct-J/K rejected shell quartets");
  } else {
    const std::uint64_t expected_full = (builds + 1U) / 2U;
    require(diagnostic.anchor_full_builds == expected_full,
            "screened incremental mode accumulated more than one delta update");
    require(diagnostic.periodic_rebuilds + 1U == diagnostic.anchor_full_builds,
            "screened incremental periodic rebuild counters disagree");
  }

  // Switch the same cached owner back to the ordinary policy. The bucket
  // identity must rebuild rather than replaying an incremental captured Graph.
  const auto ordinary_again = run_cached(&plan, system, baseline_options, unrestricted);
  require(ordinary_again.status == GENERATIVEQC_STATUS_SUCCESS && ordinary_again.scf.converged &&
              !ordinary_again.scf.incremental_direct_jk.active,
          "cached CUDA plan leaked incremental Direct-J/K policy across executions");
  generativeqc::scf::destroy_rhf_cuda_bucket_plan(plan);
}

}  // namespace

int main() {
  if (!cuda_device_available()) return 77;
  try {
    verify_case(false, 0.0, 0U);
    verify_case(true, 0.0, 0U);
    verify_case(false, 1.0e-12, 8U);
    verify_case(true, 1.0e-12, 8U);
    return EXIT_SUCCESS;
  } catch (const std::exception&) {
    return EXIT_FAILURE;
  }
}
