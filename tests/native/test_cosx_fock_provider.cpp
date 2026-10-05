#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/cosx_fock_provider.hpp"
#include "dft/cosx_reference.hpp"
#include "molecule/basis.hpp"
#include "scf/fock_build.hpp"
#include "scf/fock_prepared.hpp"

extern "C" void cosx_contraction_qualification_for_test(unsigned mask, bool unavailable);

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

generativeqc::core::System h2() {
  generativeqc::core::System system;
  system.atoms = {{1, {0.0, 0.0, 0.0}}, {1, {0.1, 0.2, 1.4}}};
  system.shells = {
      {0,
       0,
       {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}},
      {1,
       0,
       {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}},
  };
  std::string detail;
  require(
      generativeqc::molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
      "COSX provider H2 normalization failed");
  return system;
}

double max_error(const std::vector<double>& first, const std::vector<double>& second) {
  require(first.size() == second.size(), "COSX provider comparison size mismatch");
  double error = 0.0;
  for (std::size_t i = 0; i < first.size(); ++i)
    error = std::max(error, std::abs(first[i] - second[i]));
  return error;
}

generativeqc::scf::ResolvedFockBuild mixed_strategy(generativeqc::scf::FockSpin spin,
                                                    unsigned derivative_order = 0) {
  using namespace generativeqc::scf;
  auto spec = make_hf_fock_spec(spin);
  spec.derivative_order = derivative_order;
  spec.coulomb.approximation = FockApproximation::DensityFitted;
  spec.exchange.approximation = FockApproximation::SeminumericalCosx;
  spec.exchange.cosx = make_cosx_v1_spec(12, 8, 16, 3, 1.0e-12);
  return resolve_fock_build(spec, FockBackend::Cuda, 1.0e-12, 1.0e-10);
}

generativeqc::scf::ResolvedFockBuild cpu_j_strategy(
    const generativeqc::scf::ResolvedFockBuild& mixed) {
  auto spec = mixed.spec;
  spec.exchange.present = false;
  return generativeqc::scf::resolve_fock_build(spec, generativeqc::scf::FockBackend::Cpu,
                                               mixed.screening_tolerance,
                                               mixed.metric_relative_threshold);
}

void verify_admission(const generativeqc::dft::CosxFockPreparationDiagnostic& diagnostic,
                      unsigned expected_mask, std::size_t spin_builds) {
  const auto& j = diagnostic.coulomb;
  const auto& k = diagnostic.exchange;
  const auto response_peak = std::max(j.response_device_bytes, diagnostic.derivative.device_bytes);
  require(j.peak_device_bytes == j.device_bytes + j.response_device_bytes &&
              j.peak_device_bytes <= j.device_budget_bytes,
          "Coulomb diagnostic lost its unallocated response reservation");
  require(diagnostic.device_bytes == j.device_bytes + k.device_bytes &&
              diagnostic.derivative_peak_device_bytes == diagnostic.device_bytes + response_peak &&
              diagnostic.derivative_peak_device_bytes <= diagnostic.device_budget_bytes &&
              diagnostic.exchange_budget_bytes ==
                  diagnostic.device_budget_bytes - j.device_bytes - response_peak &&
              k.device_budget_bytes == diagnostic.exchange_budget_bytes &&
              diagnostic.contraction_peak_host_bytes ==
                  k.contraction_host_bytes + diagnostic.derivative.contraction_host_bytes,
          "enclosing Fock admission overlaps value/provider and response capacity");
  const auto full_tiles = k.npoint / k.tile_points;
  const auto tail_tiles = std::size_t{k.npoint % k.tile_points != 0};
  for (std::size_t slot = 0; slot < k.contractions.size(); ++slot) {
    const auto& site = k.contractions[slot];
    const bool tail = slot < 4 ? slot >= 2 : slot == 5;
    // The existing hook qualifies operations, each covering its full/tail
    // descriptors: projection bit 0, accumulation bit 1, weighted ESP bit 2.
    const bool library = (expected_mask & (1U << (slot < 4 ? slot % 2 : 2))) != 0;
    if (site.calls != spin_builds * (tail ? tail_tiles : full_tiles) ||
        site.summands != site.calls * site.resolved.summands() ||
        site.candidate.provider != (library ? "cublas" : "generated.cuda"))
      std::cerr << "mask=" << expected_mask << " slot=" << slot << " calls=" << site.calls
                << " full=" << full_tiles << " tail=" << tail_tiles << " spins=" << spin_builds
                << " work=" << site.summands << " per_call=" << site.resolved.summands()
                << " provider=" << site.candidate.provider
                << " reason=" << site.alternative_rejection << '\n';
    require(site.calls == spin_builds * (tail ? tail_tiles : full_tiles) &&
                site.summands == site.calls * site.resolved.summands() &&
                site.candidate.provider == (library ? "cublas" : "generated.cuda"),
            "enclosing Fock diagnostic lost selected full/tail provider or actual work");
  }
  require((k.provider_allowance != 0) == (expected_mask != 0) &&
              k.retained_provider_bytes <= k.provider_allowance,
          "enclosing Fock did not charge the selected shared provider once");
}

void verify_restricted(const generativeqc::core::System& system, int device,
                       unsigned expected_mask) {
  using namespace generativeqc;
  const auto strategy = mixed_strategy(scf::FockSpin::Restricted);
  bool legacy_rejected = false;
  try {
    scf::PreparedFockPlan invalid(system, &system, strategy, device);
  } catch (const std::invalid_argument&) {
    legacy_rejected = true;
  }
  require(legacy_rejected,
          "legacy PreparedFockPlan silently routed COSX through an exact/DF provider");

  dft::PreparedCosxFockPlan gpu(system, &system, strategy, 7, device);

  const std::vector<double> density{0.8, 0.2, 0.2, 0.6};
  const auto actual = gpu.build(density);

  scf::PreparedFockPlan cpu_j(system, &system, cpu_j_strategy(strategy));
  auto expected = cpu_j.build(density);
  const auto reference_k =
      dft::build_cosx_reference(system, gpu.grid().points(), gpu.grid().weights(), density,
                                dft::CosxDensityConvention::rhf_spin_summed);
  expected.exchange_alpha = reference_k.exchange;

  require(max_error(actual.coulomb, expected.coulomb) < 3.0e-10,
          "prepared RI-J differs from the CPU DF oracle");
  require(max_error(actual.exchange_alpha, expected.exchange_alpha) < 3.0e-12 &&
              actual.exchange_beta.empty(),
          "prepared COSX-K differs from the CPU RHF oracle");

  const double actual_energy = scf::contract_fock_energy(strategy, actual, density);
  const double expected_energy = scf::contract_fock_energy(strategy, expected, density);
  require(std::abs(actual_energy - expected_energy) < 3.0e-10,
          "prepared RI-J/COSX-K two-electron energy differs from the independent oracles");

  const auto actual_fock = scf::assemble_fock(strategy, gpu.one_electron().hcore, actual);
  const auto expected_fock = scf::assemble_fock(strategy, gpu.one_electron().hcore, expected);
  require(max_error(actual_fock.alpha, expected_fock.alpha) < 3.0e-10 && actual_fock.beta.empty(),
          "prepared RI-J/COSX-K RHF assembly differs from the independent oracles");

  const auto& diagnostic = gpu.diagnostic();
  verify_admission(diagnostic, expected_mask, 1);
  require(diagnostic.strategy == strategy &&
              diagnostic.coulomb.strategy.spec.exchange.present == false &&
              diagnostic.exchange.esp_on_device && diagnostic.exchange.assembly_on_device &&
              diagnostic.device_bytes <= diagnostic.device_budget_bytes &&
              diagnostic.tile_points == 7,
          "prepared COSX diagnostic lost provider identity or bounded resources");
  const auto& model = strategy.spec.exchange.cosx;
  require(gpu.grid().spec().version == model.grid_version &&
              gpu.grid().spec().radial_points == model.radial_points &&
              gpu.grid().spec().angular_polar == model.angular_polar &&
              gpu.grid().spec().angular_azimuth == model.angular_azimuth &&
              gpu.grid().spec().partition_iterations == model.partition_iterations &&
              gpu.grid().spec().coincident_tolerance == model.coincident_tolerance &&
              gpu.grid().spec().element_radii == model.element_radii,
          "prepared COSX grid does not reproduce the resolved mathematical identity");

  const auto cosx_only = dft::cuda_cosx_staging_diagnostic(system, gpu.grid().point_count(), 7);
  bool budget_rejected = false;
  try {
    dft::PreparedCosxFockPlan too_small(system, &system, strategy, 7, device,
                                        cosx_only.device_bytes);
  } catch (const std::bad_alloc&) {
    budget_rejected = true;
  }
  require(budget_rejected,
          "prepared RI-J/COSX-K accepted a budget with no capacity for the J provider");

  const auto force_strategy = mixed_strategy(scf::FockSpin::Restricted, 1);
  dft::PreparedCosxFockPlan force_gpu(system, &system, force_strategy, 7, device);
  scf::PreparedFockPlan force_cpu_j(system, &system, cpu_j_strategy(force_strategy));
  auto expected_derivative = force_cpu_j.energy_derivative(density);
  const auto reference_derivative = dft::build_cosx_molecular_derivative_reference(
      force_gpu.grid(), density, dft::CosxDensityConvention::rhf_spin_summed);
  for (std::size_t coordinate = 0; coordinate < expected_derivative.size(); ++coordinate)
    expected_derivative[coordinate] += reference_derivative.nuclear_gradient[coordinate];
  const auto components = force_gpu.energy_derivative_components(density);
  require(components.coulomb.size() == expected_derivative.size() &&
              components.exchange.size() == expected_derivative.size(),
          "prepared COSX split derivative components have the wrong shape");
  std::vector<double> recomposed(components.coulomb.size());
  for (std::size_t coordinate = 0; coordinate < recomposed.size(); ++coordinate)
    recomposed[coordinate] = components.coulomb[coordinate] + components.exchange[coordinate];
  const auto actual_derivative = force_gpu.energy_derivative(density);
  require(max_error(recomposed, actual_derivative) < 2.0e-12,
          "prepared COSX split derivative components do not recompose the public response");
  require(max_error(actual_derivative, expected_derivative) < 3.0e-8,
          "prepared RI-J/COSX-K RHF derivative differs from independent J/K oracles");
  require(force_gpu.diagnostic().derivative.bounded_tiling &&
              force_gpu.diagnostic().derivative_peak_device_bytes <=
                  force_gpu.diagnostic().device_budget_bytes,
          "prepared COSX force provider lost bounded peak-resource accounting");

  if (expected_mask) {
    // A small positive envelope still prepares the complete generated endpoint.
    // It must not borrow either provider's future derivative capacity to make a
    // qualified optional recipe fit. Include a modest independent headroom for
    // the unchanged J tile planner, well below one shared provider reservation.
    const auto tight_budget = cosx_only.device_bytes + force_gpu.diagnostic().coulomb.device_bytes +
                              force_gpu.diagnostic().derivative.device_bytes + (16U << 20);
    dft::PreparedCosxFockPlan tight(system, &system, force_strategy, 7, device, tight_budget);
    const auto tight_value = tight.build(density);
    verify_admission(tight.diagnostic(), 0, 1);
    require(max_error(tight_value.exchange_alpha, reference_k.exchange) < 3.0e-12 &&
                max_error(tight.energy_derivative(density), expected_derivative) < 3.0e-8,
            "tight enclosing Fock fallback changed value or molecular response");
  }

  auto scaled_spec = force_strategy.spec;
  scaled_spec.exchange.coefficient *= 0.5;
  const auto scaled_strategy = scf::resolve_fock_build(scaled_spec, scf::FockBackend::Cuda,
                                                       force_strategy.screening_tolerance,
                                                       force_strategy.metric_relative_threshold);
  dft::PreparedCosxFockPlan scaled_gpu(system, &system, scaled_strategy, 7, device);
  scf::PreparedFockPlan scaled_cpu_j(system, &system, cpu_j_strategy(scaled_strategy));
  auto scaled_expected = scaled_cpu_j.energy_derivative(density);
  for (std::size_t coordinate = 0; coordinate < scaled_expected.size(); ++coordinate)
    scaled_expected[coordinate] += 0.5 * reference_derivative.nuclear_gradient[coordinate];
  require(max_error(scaled_gpu.energy_derivative(density), scaled_expected) < 3.0e-8,
          "prepared COSX derivative ignored the resolved arbitrary exchange coefficient");
}

void verify_unrestricted(const generativeqc::core::System& system, int device,
                         unsigned expected_mask) {
  using namespace generativeqc;
  const auto strategy = mixed_strategy(scf::FockSpin::Unrestricted);
  dft::PreparedCosxFockPlan gpu(system, &system, strategy, 5, device);

  const std::vector<double> alpha{0.45, 0.10, 0.10, 0.35};
  const std::vector<double> beta{0.25, 0.04, 0.04, 0.18};
  const auto actual = gpu.build(alpha, beta);
  verify_admission(gpu.diagnostic(), expected_mask, 2);

  scf::PreparedFockPlan cpu_j(system, &system, cpu_j_strategy(strategy));
  auto expected = cpu_j.build(alpha, beta);
  expected.exchange_alpha =
      dft::build_cosx_reference(system, gpu.grid().points(), gpu.grid().weights(), alpha,
                                dft::CosxDensityConvention::spin_resolved)
          .exchange;
  expected.exchange_beta =
      dft::build_cosx_reference(system, gpu.grid().points(), gpu.grid().weights(), beta,
                                dft::CosxDensityConvention::spin_resolved)
          .exchange;

  require(max_error(actual.coulomb, expected.coulomb) < 3.0e-10 &&
              max_error(actual.exchange_alpha, expected.exchange_alpha) < 3.0e-12 &&
              max_error(actual.exchange_beta, expected.exchange_beta) < 3.0e-12,
          "prepared UHF RI-J/COSX-K matrices differ from the independent oracles");
  require(std::abs(scf::contract_fock_energy(strategy, actual, alpha, beta) -
                   scf::contract_fock_energy(strategy, expected, alpha, beta)) < 3.0e-10,
          "prepared UHF RI-J/COSX-K energy differs from the independent oracles");

  const auto actual_fock = scf::assemble_fock(strategy, gpu.one_electron().hcore, actual);
  const auto expected_fock = scf::assemble_fock(strategy, gpu.one_electron().hcore, expected);
  require(max_error(actual_fock.alpha, expected_fock.alpha) < 3.0e-10 &&
              max_error(actual_fock.beta, expected_fock.beta) < 3.0e-10,
          "prepared UHF RI-J/COSX-K Fock differs from the independent oracles");

  const auto force_strategy = mixed_strategy(scf::FockSpin::Unrestricted, 1);
  dft::PreparedCosxFockPlan force_gpu(system, &system, force_strategy, 5, device);
  scf::PreparedFockPlan force_cpu_j(system, &system, cpu_j_strategy(force_strategy));
  auto expected_derivative = force_cpu_j.energy_derivative(alpha, beta);
  const auto alpha_reference = dft::build_cosx_molecular_derivative_reference(
      force_gpu.grid(), alpha, dft::CosxDensityConvention::spin_resolved);
  const auto beta_reference = dft::build_cosx_molecular_derivative_reference(
      force_gpu.grid(), beta, dft::CosxDensityConvention::spin_resolved);
  for (std::size_t coordinate = 0; coordinate < expected_derivative.size(); ++coordinate)
    expected_derivative[coordinate] +=
        alpha_reference.nuclear_gradient[coordinate] + beta_reference.nuclear_gradient[coordinate];
  require(max_error(force_gpu.energy_derivative(alpha, beta), expected_derivative) < 3.0e-8,
          "prepared RI-J/COSX-K UHF derivative differs from independent J/K oracles");
}

}  // namespace

int main() {
  try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    // Qualification is below the scientific owner. Each route reuses the same
    // independent RI-J/COSX value, energy, spin and molecular-derivative oracles.
    for (unsigned mask : {0U, 1U, 2U, 3U, 4U, 7U}) {
      cosx_contraction_qualification_for_test(mask, false);
      verify_restricted(h2(), 0, mask);
    }
    for (unsigned mask : {0U, 7U}) {
      cosx_contraction_qualification_for_test(mask, false);
      verify_unrestricted(h2(), 0, mask);
    }
    cosx_contraction_qualification_for_test(7U, true);
    verify_restricted(h2(), 0, 0);
    cosx_contraction_qualification_for_test(0, false);
    std::cout << "prepared RI-J/COSX-K fixed-density RHF/UHF provider PASS\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
