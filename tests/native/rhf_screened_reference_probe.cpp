#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "molecule/basis.hpp"
#include "scf/mean_field.hpp"

namespace {
struct PlanOwner {
  generativeqc::scf::CudaRhfBucketPlan* plan{};
  ~PlanOwner() { generativeqc::scf::destroy_rhf_cuda_bucket_plan(plan); }
};

/** Emit the published physical frame, never a DIIS extrapolate or seed. */
void write_result(const generativeqc::scf::ScfResult& result, bool reused, double elapsed) {
  if (!result.converged || !result.reference)
    throw std::runtime_error("reference did not converge");
  const auto& reference = *result.reference;
  std::cout << std::setprecision(17) << "{\"energy\":" << result.energy
            << ",\"iterations\":" << result.iterations
            << ",\"refinement\":" << result.precision.refinement_iterations
            << ",\"strict_refinement\":" << result.precision.strict_refinement_applied
            << ",\"focks\":" << result.precision.strict_stage_fock_builds
            << ",\"post_focks\":" << result.precision.post_scf_fock_builds
            << ",\"work_valid\":" << result.precision.operator_work_counters_valid
            << ",\"reused\":" << reused << ",\"seconds\":" << elapsed
            << ",\"commutator\":" << reference.commutator_residual
            << ",\"canonical_drift\":" << reference.canonical_density_drift;
  const auto array = [](const char* name, const std::vector<double>& values) {
    std::cout << ",\"" << name << "\":[";
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (index) std::cout << ',';
      std::cout << values[index];
    }
    std::cout << ']';
  };
  array("density", reference.density);
  array("orbital_energies", reference.orbital_energies);
  std::cout << "}\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::invalid_argument("expected geometry-only input");
    std::ifstream input(argv[1]);
    std::size_t atoms{}, shells{};
    input >> atoms >> shells;
    generativeqc::core::System system;
    system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
    system.atoms.resize(atoms);
    system.shells.resize(shells);
    for (auto& atom : system.atoms)
      input >> atom.atomic_number >> atom.position[0] >> atom.position[1] >> atom.position[2];
    for (auto& shell : system.shells) {
      std::size_t primitives{};
      input >> shell.atom_index >> shell.angular_momentum >> primitives;
      shell.primitives.resize(primitives);
      for (auto& primitive : shell.primitives) input >> primitive.exponent >> primitive.coefficient;
    }
    if (!input) throw std::invalid_argument("truncated input");
    std::string detail;
    if (generativeqc::molecule::validate_and_normalize(system, detail) !=
        GENERATIVEQC_STATUS_SUCCESS)
      throw std::invalid_argument(detail);
    generativeqc::scf::ScfOptions options;
    options.max_iterations = 150;
    options.energy_tolerance = 1e-12;
    options.density_tolerance = 1e-11;
    options.screening_tolerance = 0;
    options.precision_mode = GENERATIVEQC_PRECISION_FP64;
    options.compute_forces = false;
    options.export_physical_reference = true;
    options.reference_memory_budget_bytes = 1ULL << 30;
    PlanOwner owner;
    std::vector<double> seed;
    for (unsigned phase = 0; phase < 5; ++phase) {
      options.reference_iteration_screening_tolerance = phase == 0 || phase == 4 ? 0 : 1e-12;
      if (phase == 3) system.atoms[1].position[2] += 0.02;
      bool reused = false;
      const auto start = std::chrono::steady_clock::now();
      auto result = generativeqc::scf::run_rhf_cuda_cached(&owner.plan, system, options, 0,
                                                           phase >= 2 ? &seed : nullptr, &reused);
      const double elapsed =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      write_result(result, reused, elapsed);
      seed = result.reference->density;
    }
    options.max_iterations = 1;
    options.reference_iteration_screening_tolerance = 1e-12;
    const auto unfinished = generativeqc::scf::run_rhf_cuda(system, options, 0);
    if (unfinished.converged || unfinished.reference)
      throw std::runtime_error("unfinished target refinement published a reference");
    for (double invalid : {-1.0, std::numeric_limits<double>::infinity()}) {
      options.reference_iteration_screening_tolerance = invalid;
      bool rejected = false;
      try {
        (void)generativeqc::scf::run_rhf_cuda(system, options, 0);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      if (!rejected) throw std::runtime_error("invalid screening policy was accepted");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
