#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "molecule/basis.hpp"
#include "scf/cuda/rhf_bucket_internal.hpp"
#include "scf/mean_field.hpp"

namespace {
using namespace generativeqc;

void require(bool accepted, const char* reason) {
  if (!accepted) throw std::runtime_error(reason);
}

struct Owner {
  scf::CudaRhfBucketPlan* plan{};
  ~Owner() { scf::destroy_rhf_cuda_bucket_plan(plan); }
};

void check(const scf::ScfResult& result, double expected) {
  require(result.converged && result.reference, "physical reference not published");
  require(std::abs(result.reference->energy - expected) < 2e-9, "independent RHF energy parity");
  require(result.reference->commutator_residual < 1e-8 &&
              result.reference->canonical_density_drift < 1e-8 &&
              result.reference->eigen_residual < 1e-8,
          "physical/canonical reference gate");
  require(result.precision.operator_work_counters_valid && result.precision.effective_bits == 64,
          "strict FP64 work provenance");
}
}  // namespace

int main(int argc, char** argv) {
  using namespace generativeqc;
  try {
    require(argc == 2, "usage: rhf-resident-values-probe INPUT");
    std::ifstream input(argv[1]);
    std::size_t atoms = 0, shells = 0;
    double original_energy = 0, displaced_energy = 0;
    input >> atoms >> shells >> original_energy >> displaced_energy;
    core::System system;
    system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
    system.atoms.resize(atoms);
    system.shells.resize(shells);
    for (auto& atom : system.atoms)
      input >> atom.atomic_number >> atom.position[0] >> atom.position[1] >> atom.position[2];
    for (auto& shell : system.shells) {
      std::size_t primitives = 0;
      input >> shell.atom_index >> shell.angular_momentum >> primitives;
      shell.primitives.resize(primitives);
      for (auto& primitive : shell.primitives) input >> primitive.exponent >> primitive.coefficient;
    }
    require(bool(input), "fixture parse");
    std::string detail;
    require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
            "fixture normalization");
    require(molecule::ao_count(system) >= 64, "fixture must exercise value admission");
    scf::ScfOptions options;
    options.compute_forces = false;
    options.export_physical_reference = true;
    options.screening_tolerance = 0;
    options.max_iterations = 100;
    options.energy_tolerance = 1e-12;
    options.density_tolerance = 1e-11;
    options.precision_mode = GENERATIVEQC_PRECISION_FP64;
    options.reference_memory_budget_bytes = 8ULL << 30;
    Owner owner;
    const auto solve_on = [&](Owner& target, const core::System& geometry,
                              const std::vector<double>* seed) {
      auto result = scf::run_rhf_cuda_cached(&target.plan, geometry, options, 0, seed);
      require(target.plan && target.plan->resources.reference_eri_ == nullptr,
              "quartet reference retained a legacy dense cache");
      return result;
    };
    const auto solve = [&](const core::System& geometry, const std::vector<double>* seed) {
      return solve_on(owner, geometry, seed);
    };
    setenv("GENERATIVEQC_RHF_RESIDENT_VALUES", "1", 1);
    const auto cold = solve(system, nullptr);
    check(cold, original_energy);
    const auto retained_bytes = scf::hf_cuda_owned_device_bytes(owner.plan);
    auto seed = cold.reference->density;
    const auto warm = solve(system, &seed);
    check(warm, original_energy);
    require(scf::hf_cuda_owned_device_bytes(owner.plan) == retained_bytes,
            "phase cache grew retained bucket storage");
    auto displaced = system;
    displaced.atoms[1].position[2] += 0.01;
    const auto moved = solve(displaced, &seed);
    check(moved, displaced_energy);
    seed = moved.reference->density;
    const auto restored = solve(system, &seed);
    check(restored, original_energy);
    setenv("GENERATIVEQC_RHF_RESIDENT_VALUES", "0", 1);
    seed = restored.reference->density;
    const auto fallback = solve(system, &seed);
    check(fallback, original_energy);
    require(scf::hf_cuda_owned_device_bytes(owner.plan) == retained_bytes,
            "optional source survived into fallback execution");
    setenv("GENERATIVEQC_RHF_RESIDENT_VALUES", "1", 1);
    // The bucket's mandatory admission excludes retired phase values. A new
    // tight-budget execution must rebuild local graphs using the exact fallback.
    options.reference_memory_budget_bytes = owner.plan->resources.reference_peak_bytes_ + (1 << 20);
    const auto tight = solve(system, &seed);
    check(tight, original_energy);
    options.reference_memory_budget_bytes = 8ULL << 30;
    Owner automatic_owner;
    unsetenv("GENERATIVEQC_RHF_RESIDENT_VALUES");
    const auto automatic_cold = solve_on(automatic_owner, system, nullptr);
    check(automatic_cold, original_energy);
    const auto automatic_retained = scf::hf_cuda_owned_device_bytes(automatic_owner.plan);
    seed = automatic_cold.reference->density;
    setenv("GENERATIVEQC_RHF_RESIDENT_VALUES", "auto", 1);
    const auto automatic_warm = solve_on(automatic_owner, system, &seed);
    check(automatic_warm, original_energy);
    require(scf::hf_cuda_owned_device_bytes(automatic_owner.plan) == automatic_retained,
            "automatic values survived the reference phase");
    const auto ordinary_generation = automatic_owner.plan->graphs.iteration_generation();
    require(automatic_owner.plan->graphs.has_iteration() && ordinary_generation != 0,
            "automatic warm refusal lacks ordinary graphs");
    const auto automatic_warm_again = solve_on(automatic_owner, system, &seed);
    check(automatic_warm_again, original_energy);
    require(automatic_owner.plan->graphs.iteration_generation() == ordinary_generation,
            "automatic warm refusal recaptured ordinary graphs");
    Owner automatic_tight_owner;
    options.reference_memory_budget_bytes =
        automatic_owner.plan->resources.reference_peak_bytes_ + (1 << 20);
    const auto automatic_tight = solve_on(automatic_tight_owner, system, nullptr);
    check(automatic_tight, original_energy);
    std::cout << "{\"passed\":true,\"nbf\":" << molecule::ao_count(system)
              << ",\"retained_device_bytes\":" << retained_bytes
              << ",\"cold_iterations\":" << cold.iterations
              << ",\"warm_iterations\":" << warm.iterations
              << ",\"automatic_cold_iterations\":" << automatic_cold.iterations
              << ",\"automatic_warm_iterations\":" << automatic_warm.iterations << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
