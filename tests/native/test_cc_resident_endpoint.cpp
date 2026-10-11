#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "cc/rccsdt_force.hpp"
#include "molecule/basis.hpp"
#include "posthf/native_provider.hpp"
#include "scf/mean_field.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

/** H2/STO-3G inputs and independent PySCF 2.14 reference from
 * tests/reference_data/cc/gradients/h2{,_shifted}.json. Changing a coordinate
 * exercises a fresh terminal state, not transport of the previous response.
 */
generativeqc::core::System molecule(bool changed) {
  generativeqc::core::System system;
  system.atoms = {{1, {0.13, -0.21, -0.67}}, {1, {-0.08, 0.17, changed ? 0.89 : 0.74}}};
  const std::vector<generativeqc::core::Primitive> primitives{
      {3.42525091, 0.1543289673}, {0.62391373, 0.5353281423}, {0.1688554, 0.4446345422}};
  system.shells = {{0, 0, primitives}, {1, 0, primitives}};
  std::string detail;
  require(
      generativeqc::molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
      "molecular fixture validation");
  return system;
}

struct Endpoint {
  double energy{}, seconds{};
  generativeqc::cc::RccsdtForceResult force;
};

/** Evaluate the complete existing E+F product, including a fresh native primal,
 * seven MO provider blocks, CC solve, Lambda/parameter/orbital and derivative
 * stages. Only the Lambda vector schedule differs between same-backend pairs.
 */
Endpoint evaluate(bool cuda, bool resident, bool changed, std::size_t budget) {
  using namespace generativeqc;
  const auto started = std::chrono::steady_clock::now();
  const auto system = molecule(changed);
  scf::ScfOptions scf_options;
  scf_options.export_physical_reference = true;
  scf_options.compute_forces = false;
  scf_options.screening_tolerance = 0.0;
  scf_options.energy_tolerance = 1e-13;
  scf_options.density_tolerance = 1e-11;
  scf_options.max_iterations = 200;
  scf_options.reference_memory_budget_bytes = budget;
  scf::ScfResult hf;
  std::shared_ptr<const integrals::ElectronInteractionSource> cuda_source;
#if GENERATIVEQC_HAS_CUDA
  if (cuda)
    hf = scf::run_rhf_cuda(system, scf_options, 0, nullptr, &cuda_source);
  else
#endif
    hf = scf::run_rhf(system, scf_options);
  require(hf.converged && hf.reference, "terminal native HF state");
  const auto& reference = *hf.reference;
  require(reference.nbf == 2 && reference.nocc == 1, "H2 orbital domain");
  std::unique_ptr<posthf::RawSource> cpu_source;
  if (cuda) {
    require(cuda_source &&
                cuda_source->supports_device_read(integrals::ElectronInteractionOperator::eri, 0),
            "CUDA endpoint requires the native reference source, not host raw ERIs");
  } else {
    cpu_source = std::make_unique<posthf::RawSource>(system);
  }
  const auto& source = cuda ? *cuda_source : *cpu_source;
  posthf::NativeBlockProvider provider(source, reference, budget, 2);
  const std::vector<std::size_t> occupied{0}, virtuals{1};
  cc::Problem problem;
  problem.nocc = problem.nvir = 1;
  problem.reference_energy = reference.energy;
  for (const auto* values :
       {&reference.overlap, &reference.hcore, &reference.fock, &reference.coefficients,
        &reference.orbital_energies, &reference.density, &reference.weighted_density})
    problem.reference_retained_bytes += values->capacity() * sizeof(double);
  problem.reference_retained_bytes += source.retained_numeric_bytes();
  problem.foo = {reference.orbital_energies[0]};
  problem.fov = {0.0};
  problem.fvv = {reference.orbital_energies[1]};
  problem.ovov = provider.get({occupied, virtuals, occupied, virtuals}, cuda);
  problem.ovvo = provider.get({occupied, virtuals, virtuals, occupied}, cuda);
  problem.oovv = provider.get({occupied, occupied, virtuals, virtuals}, cuda);
  problem.ovvv = provider.get({occupied, virtuals, virtuals, virtuals}, cuda);
  problem.ovoo = provider.get({occupied, virtuals, occupied, occupied}, cuda);
  problem.oooo = provider.get({occupied, occupied, occupied, occupied}, cuda);
  problem.vvvv = provider.get({virtuals, virtuals, virtuals, virtuals}, cuda);
  cc::SolverOptions options;
  options.max_bytes = budget;
  options.max_iterations = 150;
  options.energy_tolerance = 1e-13;
  options.residual_tolerance = 1e-11;
  cc::initialize_canonical_denominators(problem, reference.orbital_energies, options, false);
  problem.initial_t1 = {0.0};
  problem.initial_t2 = {problem.ovov[0] / problem.d2[0]};
  cc::SolverResult solved;
#if GENERATIVEQC_HAS_CUDA
  if (cuda)
    solved = cc::solve_cuda(problem, options, 0);
  else
#endif
    solved = cc::solve_cpu(problem, options);
  require(solved.converged(), "terminal native CC state");
  const std::span<const double> energies(reference.orbital_energies);
  Endpoint endpoint;
  endpoint.energy = solved.total_energy;
  if (cuda)
    endpoint.force =
        cc::rccsd_force_cuda(system, source, reference, problem, solved, energies.first(1),
                             energies.subspan(1), budget, 0, budget, resident);
  else
    endpoint.force = cc::rccsd_force_cpu(system, source, reference, problem, solved,
                                         energies.first(1), energies.subspan(1), budget);
  endpoint.seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  require(std::abs(endpoint.energy - (changed ? -1.1273910916156988 : -1.1356788968193838)) < 2e-9,
          "independent complete energy oracle");
  const std::array<double, 3> gradient =
      changed
          ? std::array<double, 3>{0.009850772911706308, -0.01782520812594464, -0.07317717020124598}
          : std::array<double, 3>{0.0050960881335866265, -0.009221492813156748,
                                  -0.03421659175408176};
  require(endpoint.force.forces.size() == 6, "complete force shape");
  for (std::size_t axis = 0; axis < 3; ++axis) {
    require(std::abs(endpoint.force.forces[axis] + gradient[axis]) < 1e-6,
            "independent atom-one force oracle");
    require(std::abs(endpoint.force.forces[axis + 3] - gradient[axis]) < 1e-6,
            "independent atom-two force oracle");
  }
  require(endpoint.force.lambda.independent_residual_norm <= 1e-9 &&
              endpoint.force.independent_orbital_residual <= 1e-9,
          "independent terminal response residuals");
  require(endpoint.force.numeric_capacity_bytes <= budget, "complete host budget");
  if (cuda)
    require(endpoint.force.lambda.resident_krylov == resident,
            "complete endpoint Lambda schedule selection");
  return endpoint;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const bool cuda = argc >= 2 && std::string_view(argv[1]) == "--cuda";
    const bool isolated = argc == 3;
    if (isolated)
      require(
          std::string_view(argv[2]) == "--baseline" || std::string_view(argv[2]) == "--resident",
          "isolated schedule argument");
    std::cout << std::setprecision(17);
    for (const auto budget : {256ULL << 20, 512ULL << 20}) {
      for (const auto state : {"cold", "warm", "changed"}) {
        const bool changed = std::string_view(state) == "changed";
        if (isolated) {
          const bool resident = std::string_view(argv[2]) == "--resident";
          const auto endpoint = evaluate(cuda, resident, changed, budget);
          std::cout << "backend=" << (cuda ? "cuda" : "cpu")
                    << " schedule=" << (resident ? "resident" : "baseline") << " state=" << state
                    << " budget=" << budget << " endpoint_seconds=" << endpoint.seconds
                    << " energy=" << endpoint.energy
                    << " lambda_iterations=" << endpoint.force.lambda.iterations
                    << " lambda_actions=" << endpoint.force.lambda.operator_actions
                    << " lambda_krylov_h2d=" << endpoint.force.lambda.krylov_h2d_bytes
                    << " lambda_krylov_d2h=" << endpoint.force.lambda.krylov_d2h_bytes
                    << " lambda_krylov_fences=" << endpoint.force.lambda.krylov_synchronizations
                    << " lambda_numeric_device_bytes=" << endpoint.force.lambda.owned_device_bytes
                    << " resident_device_admission_bytes="
                    << endpoint.force.lambda.resident_device_admission_bytes
                    << " raw_source_reads=" << endpoint.force.raw_source_reads
                    << " raw_device_source_reads=" << endpoint.force.raw_device_source_reads
                    << " raw_source_values=" << endpoint.force.raw_source_values
                    << " raw_transform_fmas=" << endpoint.force.raw_transform_fmas << '\n';
          continue;
        }
        const auto baseline = evaluate(cuda, false, changed, budget);
        const auto candidate = evaluate(cuda, cuda, changed, budget);
        require(std::abs(baseline.energy - candidate.energy) < 1e-12, "paired energy");
        double force_difference = 0.0;
        for (std::size_t index = 0; index < baseline.force.forces.size(); ++index)
          force_difference = std::max(force_difference, std::abs(baseline.force.forces[index] -
                                                                 candidate.force.forces[index]));
        require(force_difference < 1e-10, "paired complete force");
        std::cout << "backend=" << (cuda ? "cuda" : "cpu") << " state=" << state
                  << " budget=" << budget << " baseline_seconds=" << baseline.seconds
                  << " resident_seconds=" << candidate.seconds
                  << " force_difference=" << force_difference
                  << " baseline_lambda_h2d=" << baseline.force.lambda.krylov_h2d_bytes
                  << " resident_lambda_h2d=" << candidate.force.lambda.krylov_h2d_bytes
                  << " resident_lambda_d2h=" << candidate.force.lambda.krylov_d2h_bytes
                  << " resident_lambda_fences=" << candidate.force.lambda.krylov_synchronizations
                  << " resident_lambda_device_bytes=" << candidate.force.lambda.owned_device_bytes
                  << '\n';
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
