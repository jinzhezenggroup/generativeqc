#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
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
                                            bool unrestricted,
                                            const std::vector<double>* initial_density = nullptr) {
  const std::vector<generativeqc::core::System> systems{system};
  const std::vector<const std::vector<double>*> initial_densities{initial_density};
  const auto rows = unrestricted ? generativeqc::scf::run_uhf_cuda_bucket_cached(
                                       plan, systems, options, initial_densities, 0, false)
                                 : generativeqc::scf::run_rhf_cuda_bucket_cached(
                                       plan, systems, options, initial_densities, 0, false);
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

/** Independent PySCF 2.14.0 STO-3G/Bohr energy-and-force fixtures, not a native
 * full/delta self-comparison. Both spin populations and moved geometries must
 * pass the ordinary endpoint gates without changing convergence tolerances. */
void compare_pyscf(const generativeqc::scf::ScfResult& result, bool unrestricted, bool moved) {
  const double energy = unrestricted ? (moved ? -3.0238944398316963 : -3.0089750919335954)
                                     : (moved ? -2.843497533600211 : -2.840418791231528);
  const double force = unrestricted ? (moved ? 0.4815583993190885 : 0.5134714730999471)
                                    : (moved ? 0.09425952363345647 : 0.1112894685854584);
  require(result.converged && std::abs(result.energy - energy) < 1e-8,
          "CUDA incremental endpoint failed the independent PySCF energy gate");
  const std::vector<double> expected{0.0, 0.0, -force, 0.0, 0.0, force};
  require(result.forces.size() == expected.size(), "PySCF force fixture shape mismatch");
  for (std::size_t element = 0; element < expected.size(); ++element)
    require(std::abs(result.forces[element] - expected[element]) < 1e-7,
            "CUDA incremental endpoint failed the independent PySCF force gate");
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
  compare_pyscf(baseline.scf, unrestricted, false);
  compare_pyscf(incremental.scf, unrestricted, false);

  const auto& diagnostic = incremental.scf.incremental_direct_jk;
  require(diagnostic.requested && diagnostic.active,
          "native CUDA did not activate requested incremental Direct-J/K");
  const std::uint64_t builds = diagnostic.anchor_full_builds + diagnostic.delta_builds;
  require(builds == incremental.scf.iterations && diagnostic.delta_builds != 0U,
          "native CUDA incremental build counters do not match SCF iterations");
  require(diagnostic.anchor_updates == (screening_tolerance == 0.0 ? diagnostic.delta_builds : 0U),
          "native CUDA incremental anchor-update semantics changed");
  require(diagnostic.max_abs_delta_density > 0.0,
          "native CUDA incremental path did not observe a density delta");
  require(diagnostic.post_scf_full_builds != 0U,
          "native CUDA incremental path skipped strict final full rebuilds");
  std::cout << (unrestricted ? "UHF" : "RHF") << " screening=" << screening_tolerance
            << " interval=" << requested_interval << " iterations=" << incremental.scf.iterations
            << " full=" << diagnostic.anchor_full_builds << " delta=" << diagnostic.delta_builds
            << " promoted=" << diagnostic.anchor_updates
            << " final_full=" << diagnostic.post_scf_full_builds << '\n';
  require(diagnostic.quartet_work_counters_valid,
          "fixed-topology incremental Direct-J/K work counters are not valid");
  require(diagnostic.full_admitted_shell_quartets <= diagnostic.full_candidate_shell_quartets &&
              diagnostic.delta_admitted_shell_quartets <= diagnostic.delta_candidate_shell_quartets,
          "incremental Direct-J/K admitted more shell quartets than it visited");
  require(diagnostic.full_admitted_quartet_tiles >= diagnostic.full_admitted_shell_quartets &&
              diagnostic.delta_admitted_quartet_tiles >= diagnostic.delta_admitted_shell_quartets,
          "incremental Direct-J/K tile census is smaller than its admitted shell-quartet census");
  require(
      diagnostic.full_candidate_shell_quartets ==
              diagnostic.full_admitted_shell_quartets + diagnostic.full_rejected_shell_quartets &&
          diagnostic.delta_candidate_shell_quartets ==
              diagnostic.delta_admitted_shell_quartets + diagnostic.delta_rejected_shell_quartets,
      "incremental Direct-J/K candidate/admitted/rejected counters do not reconcile");

  if (screening_tolerance == 0.0) {
    require(diagnostic.anchor_full_builds == 1U && diagnostic.periodic_rebuilds == 0U,
            "unscreened exact incremental mode rebuilt unexpectedly");
    require(diagnostic.full_rejected_shell_quartets == 0U &&
                diagnostic.delta_rejected_shell_quartets == 0U,
            "unscreened incremental Direct-J/K rejected shell quartets");
  } else {
    const std::uint64_t expected_full =
        requested_interval == 0U ? 1U : (builds + requested_interval) / (requested_interval + 1U);
    require(diagnostic.anchor_full_builds >= expected_full && diagnostic.anchor_full_builds >= 3U,
            "screened incremental mode omitted cadence refreshes or full-energy closure");
    require(diagnostic.periodic_rebuilds + 1U == diagnostic.anchor_full_builds,
            "screened incremental periodic rebuild counters disagree");
  }

  // Reuse the same incremental owner at changed geometry. The per-execution
  // anchor must restart with a full-density build instead of treating the new
  // geometry as a delta from the previous integral operator.
  auto changed_system = system;
  changed_system.atoms[1].position[2] += 0.03;
  generativeqc::scf::CudaRhfBucketPlan* changed_baseline_plan = nullptr;
  const auto changed_baseline =
      run_cached(&changed_baseline_plan, changed_system, baseline_options, unrestricted);
  generativeqc::scf::destroy_rhf_cuda_bucket_plan(changed_baseline_plan);
  const auto changed_incremental =
      run_cached(&plan, changed_system, incremental_options, unrestricted);
  require(changed_baseline.status == GENERATIVEQC_STATUS_SUCCESS &&
              changed_incremental.status == GENERATIVEQC_STATUS_SUCCESS,
          "changed-geometry CUDA incremental replay failed");
  compare_final_state(changed_baseline.scf, changed_incremental.scf,
                      screening_tolerance == 0.0 ? 5.0e-10 : 5.0e-8);
  compare_pyscf(changed_baseline.scf, unrestricted, true);
  compare_pyscf(changed_incremental.scf, unrestricted, true);
  require(changed_incremental.scf.incremental_direct_jk.anchor_full_builds != 0U,
          "changed geometry reused a stale incremental Direct-J/K anchor");
  if (screening_tolerance == 0.0 && requested_interval == 0U) {
    require(changed_incremental.scf.incremental_direct_jk.anchor_full_builds == 1U,
            "unscreened changed geometry established more than one full anchor");
  }

  // Switch the same cached owner back to the ordinary policy. The bucket
  // identity must rebuild rather than replaying an incremental captured Graph.
  const auto ordinary_again = run_cached(&plan, system, baseline_options, unrestricted);
  require(ordinary_again.status == GENERATIVEQC_STATUS_SUCCESS && ordinary_again.scf.converged &&
              !ordinary_again.scf.incremental_direct_jk.active,
          "cached CUDA plan leaked incremental Direct-J/K policy across executions");
  generativeqc::scf::destroy_rhf_cuda_bucket_plan(plan);
}

void verify_density_rms_gate() {
  const auto system = asymmetric_two_center(false);
  generativeqc::scf::ScfOptions options;
  options.max_iterations = 100;
  options.energy_tolerance = 1.0e-12;
  options.density_tolerance = 1.0e-10;
  options.screening_tolerance = 1.0e-12;
  options.precision_mode = GENERATIVEQC_PRECISION_FP64;
  options.compute_forces = true;
  options.incremental_direct_jk = true;
  options.incremental_direct_jk_rebuild_interval = 8;

  generativeqc::scf::CudaRhfBucketPlan* plan = nullptr;
  const auto cadence = run_cached(&plan, system, options, false);
  require(cadence.status == GENERATIVEQC_STATUS_SUCCESS && cadence.scf.converged,
          "cadence-only incremental CUDA fixture failed");
  require(cadence.scf.incremental_direct_jk.delta_builds != 0U,
          "cadence-only incremental CUDA fixture did not exercise a delta build");

  auto adaptive_options = options;
  adaptive_options.incremental_direct_jk_density_rms_threshold = 1.0e-14;
  const auto adaptive = run_cached(&plan, system, adaptive_options, false);
  require(adaptive.status == GENERATIVEQC_STATUS_SUCCESS && adaptive.scf.converged,
          "density-gated incremental CUDA fixture failed");
  compare_final_state(cadence.scf, adaptive.scf, 5.0e-8);

  const auto& before = cadence.scf.incremental_direct_jk;
  const auto& after = adaptive.scf.incremental_direct_jk;
  require(after.anchor_full_builds + after.delta_builds == adaptive.scf.iterations,
          "adaptive incremental CUDA build counters do not match SCF iterations");
  require(after.anchor_full_builds > before.anchor_full_builds &&
              after.delta_builds < before.delta_builds,
          "late-SCF RMS gate did not replace an early delta build with a full build");

  // Reusing the same plan pointer with a different threshold must rebuild the
  // captured graph rather than retaining the prior gate scalar.
  const auto cadence_again = run_cached(&plan, system, options, false);
  require(cadence_again.status == GENERATIVEQC_STATUS_SUCCESS &&
              cadence_again.scf.incremental_direct_jk.anchor_full_builds ==
                  cadence.scf.incremental_direct_jk.anchor_full_builds &&
              cadence_again.scf.incremental_direct_jk.delta_builds ==
                  cadence.scf.incremental_direct_jk.delta_builds,
          "cached CUDA plan leaked the adaptive RMS threshold");

  generativeqc::scf::destroy_rhf_cuda_bucket_plan(plan);
}

/** No-delta reuse must preserve the independently checked force determinant,
 * including repeated resident replays; an arbitrary external seed is not proof. */
void verify_warm_full_only(bool unrestricted, bool expect_reuse = true) {
  using namespace generativeqc::scf;
  const auto system = asymmetric_two_center(unrestricted);
  ScfOptions options;
  options.energy_tolerance = 1e-12;
  options.density_tolerance = 1e-10;
  options.screening_tolerance = 1e-12;
  options.compute_forces = true;
  options.precision_mode = GENERATIVEQC_PRECISION_FP64;
  options.incremental_direct_jk = true;
  options.incremental_direct_jk_rebuild_interval = 8;

  CudaRhfBucketPlan* plan = nullptr;
  const auto report_phase = [&](const char* phase, const RhfBucketItem& point) {
    std::cout << (unrestricted ? "UHF" : "RHF") << " warm-test=" << phase
              << " reuse=" << expect_reuse << " iterations=" << point.scf.iterations
              << " energy=" << std::setprecision(17) << point.scf.energy
              << " full=" << point.scf.incremental_direct_jk.anchor_full_builds
              << " delta=" << point.scf.incremental_direct_jk.delta_builds
              << " final_full=" << point.scf.precision.post_scf_fock_builds << std::endl;
  };
  const auto cold = run_cached(&plan, system, options, unrestricted);
  report_phase("cold", cold);
  require(cold.status == GENERATIVEQC_STATUS_SUCCESS, "incremental warm seed failed");
  compare_pyscf(cold.scf, unrestricted, false);
  const auto retained_density = cold.scf.density;
  for (unsigned repeat = 0; repeat < (expect_reuse ? 3U : 1U); ++repeat) {
    const auto warm = run_cached(&plan, system, options, unrestricted, &retained_density);
    report_phase("resident", warm);
    require(warm.status == GENERATIVEQC_STATUS_SUCCESS && warm.scf.initial_density_used,
            "resident incremental warm replay failed");
    compare_pyscf(warm.scf, unrestricted, false);
    compare_final_state(cold.scf, warm.scf, 1e-9);
    if (!expect_reuse) {
      CudaDirectFinalStateAudit audit;
      require(warm.scf.incremental_direct_jk.delta_builds == 0 &&
                  warm.scf.precision.post_scf_fock_builds > 0 &&
                  get_rhf_cuda_final_state_audit(plan, audit) &&
                  audit.route == CudaDirectFinalStateRoute::canonical_fallback &&
                  audit.fallback_reason ==
                      CudaDirectFinalStateFallbackReason::explicit_final_fock_rebuild,
              "full-only warm reuse ignored the explicit final rebuild diagnostic");
      destroy_rhf_cuda_bucket_plan(plan);
      return;
    }
    require(warm.scf.density == retained_density,
            "no-delta reuse advanced the certified density generation");
    require(warm.scf.incremental_direct_jk.active &&
                warm.scf.incremental_direct_jk.delta_builds == 0 &&
                warm.scf.incremental_direct_jk.post_scf_full_builds == 0 &&
                warm.scf.precision.post_scf_fock_builds == 0,
            "full-only resident warm solve performed unnecessary final Fock builds");
    CudaDirectFinalStateAudit audit;
    require(get_rhf_cuda_final_state_audit(plan, audit) &&
                audit.route == CudaDirectFinalStateRoute::scf_force_ready &&
                audit.seed_provenance && audit.physical_residual_validated &&
                audit.orbital_frame_bound && audit.restart_same_density_generation &&
                audit.additional_physical_fock_builds == 0,
            "no-delta warm publication lost its physical-state proof");
  }

  CudaRhfBucketPlan* external_plan = nullptr;
  const auto external =
      run_cached(&external_plan, system, options, unrestricted, &retained_density);
  report_phase("external", external);
  require(external.status == GENERATIVEQC_STATUS_SUCCESS,
          "external warm density failed its bounded finalization");
  compare_pyscf(external.scf, unrestricted, false);
  CudaDirectFinalStateAudit external_audit;
  require(get_rhf_cuda_final_state_audit(external_plan, external_audit) &&
              external_audit.route == CudaDirectFinalStateRoute::canonical_fallback &&
              !external_audit.seed_provenance && external.scf.precision.post_scf_fock_builds > 0,
          "external warm seed bypassed the required physical rebuild");
  destroy_rhf_cuda_bucket_plan(external_plan);

  auto moved_system = system;
  moved_system.atoms[1].position[2] += 0.03;
  const auto moved = run_cached(&plan, moved_system, options, unrestricted, &retained_density);
  report_phase("moved", moved);
  require(moved.status == GENERATIVEQC_STATUS_SUCCESS,
          "moved warm density failed its bounded finalization");
  compare_pyscf(moved.scf, unrestricted, true);
  CudaDirectFinalStateAudit moved_audit;
  require(get_rhf_cuda_final_state_audit(plan, moved_audit) &&
              moved_audit.route == CudaDirectFinalStateRoute::canonical_fallback &&
              !moved_audit.seed_provenance && moved.scf.precision.post_scf_fock_builds > 0,
          "changed geometry reused an unproven force determinant");
  destroy_rhf_cuda_bucket_plan(plan);
}

}  // namespace

int main() {
  if (!cuda_device_available()) return 77;
  try {
    require(::setenv("GENERATIVEQC_FINAL_FOCK_REBUILD", "0", 1) == 0,
            "could not select retained final-state qualification");
    verify_case(false, 0.0, 0U);
    verify_case(true, 0.0, 0U);
    verify_case(false, 1.0e-12, 8U);
    verify_case(true, 1.0e-12, 8U);
    verify_case(false, 1.0e-12, 0U);
    verify_case(true, 1.0e-12, 0U);
    verify_case(false, 1.0e-12, 1U);
    verify_case(true, 1.0e-12, 1U);
    verify_density_rms_gate();
    verify_warm_full_only(false);
    verify_warm_full_only(true);
    require(::setenv("GENERATIVEQC_FINAL_FOCK_REBUILD", "1", 1) == 0,
            "could not select explicit final rebuild qualification");
    verify_warm_full_only(false, false);
    verify_warm_full_only(true, false);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
